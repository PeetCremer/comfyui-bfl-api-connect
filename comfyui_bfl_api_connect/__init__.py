import argparse

from enum import Enum
from pydantic import BaseModel, Field, NonNegativeInt
import webbrowser
import httpx
import time
import random


class ImageVariant(str, Enum):
    dev = "flux-dev"
    pro = "flux-pro"
    proplus = "flux-pro-1.1"


class AsyncResponse(BaseModel):
    id: str


class HTTPValidationError(BaseModel):
    class HTTPValidationErrorDetail(BaseModel):
        loc: list[str | int]
        msg: str
        type: str

    detail: HTTPValidationErrorDetail


class StatusResponse(str, Enum):
    TaskNotFound = "Task not found"
    Pending = "Pending"
    RequestModerated = "Request Moderated"
    ContentModerated = "Content Moderated"
    Ready = "Ready"
    Error = "Error"


class ResultResponse(BaseModel):
    id: str
    status: StatusResponse
    result: dict | None


class ImageRequest(BaseModel):
    prompt: str = "ein fantastisches bild"
    width: int = Field(1024, ge=256, le=1440, multiple_of=32)
    height: int = Field(1024, ge=256, le=1440, multiple_of=32)
    steps: int | None = None
    prompt_upsampling: bool = False
    seed: NonNegativeInt | None = None
    guidance: float | None = Field(None, ge=1.5, le=5.0)
    safety_tolerance: int | None = Field(2, ge=0, le=6)
    interval: int | None = Field(None, ge=1, le=4)


class ValidationError(BaseModel):
    loc: list[str]
    msg: str
    type: str


def pretty_dict_str(d: dict) -> str:
    import json

    return json.dumps(d, sort_keys=True, indent=4)


def run_flux(
    api_key: str,
    image_request_body: ImageRequest,
    variant: ImageVariant = ImageVariant.proplus,
) -> None:
    bfl_url = f"https://api.bfl.ml/v1/{str(variant.value)}"
    print(
        f"Posting job to {bfl_url}:\n{pretty_dict_str(image_request_body.model_dump())}\n"
    )
    res = httpx.post(
        bfl_url,
        headers={"x-key": api_key},
        json=image_request_body.model_dump(),
    )
    res.raise_for_status()
    async_response = AsyncResponse(**res.json())
    job_id = async_response.id

    n = 1  # exponential backoff counter. For now not used.
    while True:
        # wait with exponential backoff
        time.sleep(0.5 * (2**n) + (random.randint(0, 1000) / 1000))
        # fetch result
        print(f"Fetching status of job {job_id} ...")
        res = httpx.get(
            "https://api.bfl.ml/v1/get_result",
            headers={"x-key": api_key},
            params={"id": job_id},
        )
        res.raise_for_status()
        result_response = ResultResponse(**res.json())
        match result_response.status:
            case StatusResponse.Ready:
                print(f"Result ready:\n{result_response.result}")
                assert result_response.result is not None
                sample_url = result_response.result.get("sample")
                assert sample_url is not None
                webbrowser.open(sample_url, new=0, autoraise=True)
                return
            case StatusResponse.Error:
                print(f"Error: {result_response.result}")
                return
            case StatusResponse.Pending:
                print("Job still pending ...")
                pass
            case StatusResponse.RequestModerated:
                print("Request moderated ...")
                return
            case StatusResponse.ContentModerated:
                print("Content moderated ...")
                return
            case StatusResponse.TaskNotFound:
                print("Task not found ...")
                return
            case _:
                raise ValueError(f"Unknown status: {result_response.status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", type=str, required=True, help="Black Forest Labs API key"
    )
    parser.add_argument("--prompt", type=str, default="ein fantastisches bild")
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the image (256-1440, multiple of 32)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the image (256-1440, multiple of 32)",
    )
    parser.add_argument(
        "--variant",
        type=ImageVariant,
        choices=list(ImageVariant),
        default=ImageVariant.proplus,
        help="Image variant",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps for image generation"
    )
    parser.add_argument(
        "--prompt_upsampling", action="store_true", help="Enable prompt upsampling"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random number generation"
    )
    parser.add_argument(
        "--guidance", type=float, default=None, help="Guidance value (1.5-5.0)"
    )
    parser.add_argument(
        "--safety_tolerance", type=int, default=2, help="Safety tolerance level (0-6)"
    )
    parser.add_argument(
        "--interval", type=int, default=None, help="Interval value (1-4)"
    )

    args = parser.parse_args()
    image_request_input = ImageRequest(**vars(args))

    run_flux(args.api_key, image_request_input, variant=args.variant)


if __name__ == "__main__":
    main()
