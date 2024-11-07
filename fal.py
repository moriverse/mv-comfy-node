import fal_client
import typing as t


def generate(prompt: str) -> t.Optional[str]:
    def on_queue_update(update):
        pass

    result = fal_client.subscribe(
        "fal-ai/flux-pro/v1.1-ultra",
        arguments={
            "prompt": prompt,
            "aspect_ratio": "3:4",
        },
        on_queue_update=on_queue_update,
    )

    images = result.get("images")
    if not images:
        return None

    image = images[0]
    return image.get("url")
