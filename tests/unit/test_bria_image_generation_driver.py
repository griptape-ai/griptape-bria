from griptape.bria.drivers.bria_image_generation_driver import BriaImageGenerationDriver


class TestBriaImageGenerationDriver:
    def test_init(self):
        assert BriaImageGenerationDriver(model="foo", api_key="bar")
