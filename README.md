# Griptape Bria Extension

## Overview
This extension provides an [Image Generation Driver](https://docs.griptape.ai/stable/griptape-framework/drivers/image-generation-drivers) for [Bria](https://bria.ai/).

```python
from griptape.bria.drivers.bria_image_generation_driver import BriaImageGenerationDriver
from griptape.engines import PromptImageGenerationEngine
from griptape.structures import Agent
from griptape.tools import FileManagerTool, PromptImageGenerationTool

agent = Agent(
    tools=[
        PromptImageGenerationTool(
            engine=PromptImageGenerationEngine(
                image_generation_driver=BriaImageGenerationDriver(model="2.3")
            ),
            off_prompt=True,
        ),
        FileManagerTool(),
    ]
)

agent.run(
    "Save a picture of a watercolor painting of a dog riding a skateboard to the desktop."
)
```

## Installation

Poetry:
```bash
poetry add https://github.com/griptape-ai/griptape-bria.git
```

Pip:
```bash
pip install git+https://github.com/griptape-ai/griptape-bria.git
```
