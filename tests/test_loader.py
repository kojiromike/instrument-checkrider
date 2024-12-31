from unittest import TestCase

from smitty import faa_loader


class TestFaaLoader(TestCase):
    def test_loader(self):
        self.assertTrue(faa_loader)
