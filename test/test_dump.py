import unittest

from Lonl import Lonl
from Lonl.base import DictionaryTyping, TokenType, LonlDictionary


class test_dump(unittest.TestCase):

    def test_dump_string(self):
        variable_dict = {
            'Variable1': 19, 'Variable2': 'Hello, world!', 'Variable3': True, 'Variable4': 3.999,
            'Variable5': {"Hello": "Json", "Nested": {"Hi": "world"}},
            'Variable6': ['Hello', 'World!', 'Test123']
        }
        typehint_dict = {
            'Variable1': DictionaryTyping(type=int, type2=None, tokentype=None, tokentype2=None, value=None),
            'Variable2': DictionaryTyping(type=str, type2=None, tokentype=None, tokentype2=None, value=None),
            'Variable3': DictionaryTyping(type=bool, type2=None, tokentype=None, tokentype2=None, value=None),
            'Variable4': DictionaryTyping(type=float, type2=None, tokentype=None, tokentype2=None, value=None),
            'Variable5': DictionaryTyping(type=str, type2=None, tokentype=TokenType.JSON, tokentype2=None, value=None),
            'Variable6': DictionaryTyping(type=list, type2=str, tokentype=None, tokentype2=None, value=None)
        }

        lonl_dict = LonlDictionary(variable_dict, typehint_dict)

        result = "Variable1<int> = 19;\nVariable2<str> = 'Hello, world!';\nVariable3<bool> = True;\nVariable4<float> " \
                 "= 3.999;\nVariable5<json> = {\"Hello\": \"Json\", \"Nested\": {\"Hi\": " \
                 "\"world\"}};\nVariable6<list<str>> = ['Hello', 'World!', 'Test123'];\n"

        lonl_string = Lonl.dumps(lonl_dict)

        self.assertEqual(lonl_string, result, "Should always produce the same output")

        self.assertEqual(
            Lonl.dumps(variable_dict), result, "Should produce the same output without typehint dictionary"
        )


if __name__ == '__main__':
    unittest.main()
