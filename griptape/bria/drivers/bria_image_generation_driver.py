from __future__ import annotations
import os
from PIL import Image
from io import BytesIO

from urllib.parse import urljoin
from griptape.artifacts import ImageArtifact
import requests

from attrs import define, field, Factory

from griptape.drivers import BaseImageGenerationDriver


@define
class BriaImageGenerationDriver(BaseImageGenerationDriver):
    base_url: str = field(
        default="https://engine.prod.bria-api.com",
        kw_only=True,
        metadata={"serializable": False},
    )
    api_key: str | None = field(
        default=Factory(lambda: os.environ["BRIA_API_KEY"]),
        kw_only=True,
        metadata={"serializable": False},
    )
    headers: dict[str, str] = field(
        default=Factory(
            lambda self: {
                "Content-Type": "application/json",
                "api_token": self.api_key,
            },
            takes_self=True,
        ),
        kw_only=True,
        metadata={"serializable": False},
    )
    extra_params: dict[str, str] = field(
        default=Factory(dict),
        kw_only=True,
        metadata={"serializable": False},
    )

    def try_text_to_image(
        self, prompts: list[str], negative_prompts: list[str] | None = None
    ) -> ImageArtifact:
        url = urljoin(self.base_url, f"/v1/text-to-image/base/{self.model}")
        prompt = " ".join(prompts)
        negative_prompt = " ".join(negative_prompts) if negative_prompts else ""

        payload = {
            "prompt": prompt,
            "num_results": 1,
            "sync": True,
            "negative_prompt": negative_prompt,
            **self.extra_params,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        url = response.json()["result"][0]["urls"][0]

        image_response = requests.get(url)
        image_bytes = image_response.content

        width, height = Image.open(BytesIO(image_bytes)).size

        return ImageArtifact(
            value=image_bytes, format="jpeg", width=width, height=height
        )

    def try_image_variation(
        self,
        prompts: list[str],
        image: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image variation"
        )

    def try_image_inpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inpainting"
        )

    def try_image_outpainting(
        self,
        prompts: list[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts: list[str] | None = None,
    ) -> ImageArtifact:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support outpainting"
        )
