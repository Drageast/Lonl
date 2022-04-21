import unittest

from Lonl import Lonl
from Lonl.base import LonlException


class test_load(unittest.TestCase):

    def test_load_file(self):
        with open("files/Test.lonl", "r") as f:
            data = Lonl.load(f, safe=False)

        # see if some inputs are correct
        self.assertEqual(data["Variable1"], 19, "Variable should be 19")

        # should fail because of the save mode
        with open("files/Test.lonl", "r") as f:
            self.assertRaises(LonlException.SafeModeEnabled, Lonl.load, f)

        string_lonl = \
            """
            Variable1<int> = 19;
            Variable2<str> = "Hello, world!";
            Variable3<bool> = True;
            Variable4<float> = 3.999;
            """

        data2 = Lonl.loads(string_lonl)

        self.assertEqual(data2["Variable3"], True, "Variable should be loaded and True")


if __name__ == '__main__':
    unittest.main()
