import unittest
from utils import build_model


class UtilsTest(unittest.TestCase):
    def test_build_model(self):
        model = build_model('vgg16', [4096, 1000, 512])
        print(model.classifier)
        self.assertTrue(model)
