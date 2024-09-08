import argparse

from enum import Enum
from pydantic import BaseModel, Field, NonNegativeInt


class ImageVariant(str, Enum):
    dev = "flux.1-dev"
    pro = "flux.1-pro"


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
    variant: ImageVariant = ImageVariant.pro
    steps: int | None = None
    prompt_upsampling: bool = False
    seed: NonNegativeInt | None = None
    guidance: float | None = Field(None, ge=1.5, le=5.0)
    safety_tolerance: int | None = Field(2, ge=0, le=6)
    interval: int | None = Field(None, ge=1, le=4)


class ImageResponse(BaseModel):
    result: AsyncResponse | HTTPValidationError


class ValidationError(BaseModel):
    loc: list[str]
    msg: str
    type: str


class APIKeyHeader(BaseModel):
    x_key: str


class GetResultRequest(BaseModel):
    id: str


class GetResultResponse(BaseModel):
    result: ResultResponse | HTTPValidationError


def run_flux(api_key: str, image_request_body: ImageRequest) -> None:
    print(image_request_body)


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
        default=ImageVariant.pro,
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

    run_flux(args.api_key, image_request_input)


if __name__ == "__main__":
    main()
