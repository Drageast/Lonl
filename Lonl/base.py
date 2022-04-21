# /// Imports \\\
from typing import Dict, Any, List, Optional, Set, TextIO, Tuple, Union, Type, IO
from dataclasses import dataclass
from enum import Enum

import xsync
import os
from json import loads as js_load_str
from json import load as js_load


# /// Data Structures \\\
class _TokenType(Enum):
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"

    JSON = "json"
    LIST = "list"

    FILE = "file"
    ENVIRONMENT = "env"

    LPAREN = "("
    RPAREN = ")"
    LSQUAREBR = "["
    RSQUAREBR = "]"
    LCURLYBR = "{"
    RCURLYBR = "}"

    ASTERISK = "*"
    ATSIGN = "@"
    COMMA = ","
    DOT = "."
    SEMICOLON = ";"
    COLON = ":"
    EQUALS = "="
    QUESTION = "?"
    EXCLAMATION = "!"

    LARROW = "<"
    RARROW = ">"

    WHITESPACE = " \t"
    NEWLINE = "\n"

    # Identifier
    IDENTIFIER = "Identifier   "
    # END OF FILE
    EOF = "EndOfFile   "


@dataclass
class _Token:
    type: _TokenType
    value: Optional[Any] = None
    position: Optional[Tuple[Tuple[int, int], int]] = ((0, 0), 0)  # (Start, Stop), Line

    def __repr__(self):
        return f"Token: (type:{self.type}, value:{self.value}, position:{self.position})"


@dataclass
class _DictionaryTyping:
    type: Type
    type2: Optional[Type] = None
    tokentype: Optional[_TokenType] = None
    value: Optional[Any] = None


class LonlDictionary:
    """
    This is a modified dictionary that also saves the typing definition for each key-value pair so you cant pass in
    the wrong types for each key and to remember <env> or <file> attributes to write them back correspondingly
    """
    __slots__ = ("__dict__", "_typing")

    def __init__(self, value_dict: Dict[Any, Any], typing_dict: Optional[Dict[Any, _DictionaryTyping]] = None) -> None:
        """
        Initializes the lonlDictionary class.

        :param value_dict: Takes in a normal dictionary with the key (Variable name) and the value (Variable value)
        :type value_dict: Dict[AnyStr, Any]
        :param typing_dict: Takes in a normal dictionary with the key (Variable name) and the value
            (Token with variable type and information) - If not present, types will be inferred automatically
        :type typing_dict: Optional[Dict[AnyStr, DictionaryTyping]]
        """
        for key, value in value_dict.items():
            self.__dict__[key] = value

        if not typing_dict:
            d = dict()

            for key, value in self.__dict__.items():
                d[key] = _DictionaryTyping(type=type(value))

            self._typing = d

        else:
            self._typing = typing_dict

    def __setitem__(self, key, item):
        if key in self._typing:
            if not isinstance(item, self._typing[key].type):
                raise TypeError(
                    f"Expected value for '{key}' "
                    f"to be of type '{self._typing[key].type.__name__}' - not '{type(item).__name__}'"
                )
            else:
                if isinstance(self._typing[key].type, list) and isinstance(item, list):
                    type2 = self._typing[key].type2
                    for i in item:
                        if not isinstance(i, type2):
                            raise TypeError(
                                f"Expected value for '{key}' "
                                f"to be of type '{self._typing[key].type2.__name__}' - not '{type(item[0]).__name__}'"
                            )

        elif key not in self._typing:
            if isinstance(self._typing[key].type, list) and isinstance(item, list):
                self._typing[key] = _DictionaryTyping(type=type(item), type2=type(item[0]))
            else:
                self._typing[key] = _DictionaryTyping(type=type(item))

        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self._typing[key]
        del self.__dict__[key]

    def clear(self):
        self._typing.clear()
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self.__setitem__(key, value)
        return self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        self._typing.pop(*args)
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__dict__.__cmp__(dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


# // Exceptions \\
class LonlException:
    class SafeModeEnabled(Exception):
        pass

    class Lexer:
        class LexingException(Exception):
            pass

    class Parser:
        class ParsingException(Exception):
            pass


# // Iterator and Buffer \\
class _CustomIterator:
    def __init__(self, iterable: Union[List, Tuple, Set, str, int]):
        self.__index: int = 0
        self.real_index: int = 0
        self.__reversed = False
        if isinstance(iterable, list) or isinstance(iterable, tuple) or isinstance(iterable, set):
            self.__iterable: Union[List, Tuple, Set] = iterable
        else:
            self.__iterable: Union[List, Tuple, Set] = list(iterable)

    def __iter__(self):
        return self

    async def __aiter__(self):
        return self

    def __next__(self):
        try:
            result = self.__iterable[self.__index]
        except IndexError:
            raise StopIteration
        self.__index += -1 if self.__reversed else + 1
        self.real_index += -1 if self.__reversed else + 1
        return result

    async def __anext__(self):
        try:
            result = self.__iterable[self.__index]
        except IndexError:
            raise StopIteration
        self.__index += -1 if self.__reversed else + 1
        self.real_index += -1 if self.__reversed else + 1
        return result

    def __getitem__(self, index):
        return self.__iterable[index]

    def __reversed__(self):
        self.__reversed = True
        return self

    def __len__(self):
        return len(self.__iterable)

    @property
    def index(self):
        return self.real_index

    @xsync.as_hybrid()
    def peek(self, n: int = 1):
        try:
            if self.__reversed:
                result = self.__iterable[self.__index - n - 1]
            else:
                result = self.__iterable[self.__index + n - 1]
        except IndexError:
            return None
        return result

    @xsync.set_async_impl(peek)
    async def async_peek(self, n: int = 1):
        try:
            if self.__reversed:
                result = self.__iterable[self.__index - n - 1]
            else:
                result = self.__iterable[self.__index + n - 1]
        except IndexError:
            return None
        return result

    @xsync.as_hybrid()
    def advance(self, n: int = 1):
        for _ in range(n - 1):
            try:
                self.__next__()
            except StopIteration:
                return None
        try:
            return self.__next__()
        except StopIteration:
            return None

    @xsync.set_async_impl(advance)
    async def async_advance(self, n: int = 1):
        for _ in range(n - 1):
            try:
                await self.__anext__()
            except StopAsyncIteration:
                return None
        try:
            return await self.__anext__()
        except StopAsyncIteration:
            return None


class _Buffer:
    def __init__(self):
        # position
        self.__word_start: int = 0
        self.line = 1

        # buffer variables
        self.v: str = ""
        self.t: Union[_TokenType, None] = None

    def position(self, index: int):
        return (self.__word_start, index), self.line

    def exception_pos(self, index: int, shift: int = 2):
        return self.__word_start + shift, index + shift, self.line

    def tokenize(self, index: int) -> _Token:
        tok = _Token(type=self.t, value=self.v, position=self.position(index))
        self.__word_start = index

        # reset the buffer
        self.v = ""
        self.t = None
        return tok

    def reset(self, index: int):
        self.__word_start = index

        # reset the buffer
        self.v = ""
        self.t = None

    def reset_pos(self, index: int = 0):
        self.__word_start = index


# /// Lonl implementation \\\
class Lonl:
    """
    This is the class representation for the `.lonl` language. It includes a sync and async read method for a
    filestream and a read from string method.It furthermore includes a sync and async dump method for a filestream
    and a dump string method.
    """

    @staticmethod
    def __exception_pos(pos: Tuple[Tuple[int, int], int], shift: int = 0):
        x, line = pos
        start, stop = x
        return start + shift, stop + shift, line

    @staticmethod
    def __str_to_list(value: str) -> List[Any]:
        raw_list = [x for x in value.replace("[", "").replace("]", "").replace(", ", ",").replace(
            "\"",
            ""
        ).replace("'", "")
            .split(
            ","
        )]
        return raw_list

    @staticmethod
    @xsync.as_hybrid()
    def __lex(contents: str) -> List[_Token]:
        """
        Synchronous implementation for the lexer. This function turns the provided contents into tokens
        :param contents: contents in form of a string, lonl-Language compliant
        :type contents: str
        :return: List[_Token]
        """
        # creating a custom iterable and buffer
        buffer = _Buffer()
        iterator = _CustomIterator(contents)

        # creating 2 lists with TokenType and TokenType value
        TD = [TD for TD in _TokenType]
        TDv = [TD.value for TD in _TokenType]

        # creating a empty list for the tokens
        tokens: List[_Token] = []

        # creating te current character
        cchar = next(iterator)

        while cchar is not None:
            # string
            if cchar in ["\"", "'"]:
                temp = cchar
                if buffer.t or buffer.v:
                    tokens.append(buffer.tokenize(iterator.index))

                buffer.v += cchar
                buffer.t = _TokenType.STRING
                cchar = iterator.advance()

                while cchar != temp and cchar is not None:
                    buffer.v += cchar
                    cchar = iterator.advance()

                if cchar:
                    buffer.v += cchar

                tokens.append(buffer.tokenize(iterator.index))

            # Integer, float
            elif cchar.isdigit() and not buffer.v:
                buffer.v += cchar
                buffer.t = _TokenType.INTEGER
                cchar = iterator.advance()

                while cchar.isdigit() or cchar == ".":
                    if cchar == "." and not buffer.t == _TokenType.FLOAT:
                        buffer.v += cchar
                        buffer.t = _TokenType.FLOAT

                    elif cchar.isdigit():
                        buffer.v += cchar

                    elif cchar == "." and buffer.t == _TokenType.FLOAT:
                        buffer.v += cchar
                        buffer.t = _TokenType.IDENTIFIER
                        start, stop, line = buffer.exception_pos(iterator.index)
                        raise LonlException.Lexer.LexingException(
                            f"Error lexing line: '{line}', region (characters): '"
                            f"{start}' to `{stop}`"
                        )

                    cchar = iterator.advance()

                tokens.append(buffer.tokenize(iterator.index))
                iterator.real_index += 1
                tokens.append(_Token(type=_TokenType.SEMICOLON, position=buffer.position(iterator.index)))

            # json
            elif cchar == "{":
                buffer.v += cchar
                buffer.t = _TokenType.JSON
                cchar = iterator.advance()

                while cchar != ";" and cchar is not None:
                    buffer.v += cchar
                    cchar = iterator.advance()

                iterator.real_index -= 1
                tokens.append(buffer.tokenize(iterator.index))
                iterator.real_index += 1

                tokens.append(_Token(type=_TokenType.SEMICOLON, position=buffer.position(iterator.index)))

            # list
            elif cchar == "[":
                buffer.v += cchar
                buffer.t = _TokenType.LIST
                cchar = iterator.advance()

                while cchar != ";" and cchar is not None:
                    buffer.v += cchar
                    cchar = iterator.advance()

                iterator.real_index -= 1
                tokens.append(buffer.tokenize(iterator.index))
                iterator.real_index += 1

                tokens.append(_Token(type=_TokenType.SEMICOLON, position=buffer.position(iterator.index)))

            elif cchar.isspace():
                if cchar == "\n":
                    if buffer.v:
                        iterator.real_index -= 1
                        tokens.append(buffer.tokenize(iterator.index))
                        iterator.real_index += 1

                    tokens.append(_Token(type=_TokenType.NEWLINE, position=buffer.position(iterator.index)))
                    buffer.reset(iterator.index)
                    buffer.line += 1
                    buffer.reset_pos()
                    iterator.real_index = 0

                elif cchar in [" ", "\t"]:
                    if buffer.v:
                        iterator.real_index -= 1
                        tokens.append(buffer.tokenize(iterator.index))
                        iterator.real_index += 1

                    tokens.append(_Token(type=_TokenType.WHITESPACE, position=buffer.position(iterator.index)))
                    buffer.reset(iterator.index)

            elif cchar in TDv:
                for TypeDefVar, TypeDef in zip(TDv, TD):
                    if TypeDefVar == cchar:
                        if buffer.v:
                            if cchar == ".":
                                buffer.v += "."
                                buffer.t = _TokenType.FILE
                                break
                            else:
                                iterator.real_index -= 1
                                tokens.append(buffer.tokenize(iterator.index))
                                iterator.real_index += 1

                        tokens.append(_Token(type=TypeDef, position=buffer.position(iterator.index)))
                        buffer.reset(iterator.index)
                        break

            else:
                if cchar is not None:
                    buffer.v += cchar
                    buffer.t = _TokenType.IDENTIFIER if buffer.t is None else buffer.t

                    if buffer.v.lower() in ["true", "false"] and not str(iterator.peek()).isalpha():
                        buffer.t = _TokenType.BOOLEAN
                        tokens.append(buffer.tokenize(iterator.index))

            cchar = iterator.advance()

        return tokens

    @staticmethod
    @xsync.set_async_impl(__lex)
    async def __async_lex(contents: str) -> List[_Token]:
        """
        Asynchronous implementation for the lexer. This function turns the provided contents into tokens
        :param contents: contents in form of a string, lonl compliant
        :type contents: str
        :return: List[_Token]
        """
        # creating a custom iterable and buffer
        buffer = _Buffer()
        iterator = _CustomIterator(contents)

        # creating 2 lists with TokenType and TokenType value
        TD = [TD for TD in _TokenType]
        TDv = [TD.value for TD in _TokenType]

        # creating a empty list for the tokens
        tokens: List[_Token] = []

        # creating te current character
        cchar = next(iterator)

        while cchar is not None:
            # string
            if cchar in ["\"", "'"]:
                temp = cchar
                if buffer.t or buffer.v:
                    tokens.append(buffer.tokenize(iterator.index))

                buffer.v += cchar
                buffer.t = _TokenType.STRING
                cchar = await iterator.advance()

                while cchar != temp and cchar is not None:
                    buffer.v += cchar
                    cchar = await iterator.advance()

                if cchar:
                    buffer.v += cchar

                tokens.append(buffer.tokenize(iterator.index))

            # Integer, float
            elif cchar.isdigit() and not buffer.v:
                buffer.v += cchar
                buffer.t = _TokenType.INTEGER

                while cchar.isdigit() or cchar == ".":
                    if cchar == "." and not buffer.t == _TokenType.FLOAT:
                        buffer.v += cchar
                        buffer.t = _TokenType.FLOAT

                    elif cchar.isdigit():
                        buffer.v += cchar

                    elif cchar == "." and buffer.t == _TokenType.FLOAT:
                        buffer.v += cchar
                        buffer.t = _TokenType.IDENTIFIER
                        start, stop, line = buffer.exception_pos(iterator.index)
                        raise LonlException.Lexer.LexingException(
                            f"Error lexing line: '{line}', region (characters): '"
                            f"{start}' to `{stop}`"
                        )

                    cchar = await iterator.advance()

                tokens.append(buffer.tokenize(iterator.index))

            # json
            elif cchar == "{":
                buffer.v += cchar
                buffer.t = _TokenType.JSON
                cchar = await iterator.advance()

                while cchar != ";" and cchar is not None:
                    buffer.v += cchar
                    cchar = await iterator.advance()

                iterator.real_index -= 1
                tokens.append(buffer.tokenize(iterator.index))
                iterator.real_index += 1

                tokens.append(_Token(type=_TokenType.SEMICOLON, position=buffer.position(iterator.index)))

            # list
            elif cchar == "[":
                buffer.v += cchar
                buffer.t = _TokenType.LIST
                cchar = await iterator.advance()

                while cchar != ";" and cchar is not None:
                    buffer.v += cchar
                    cchar = await iterator.advance()

                iterator.real_index -= 1
                tokens.append(buffer.tokenize(iterator.index))
                iterator.real_index += 1

                tokens.append(_Token(type=_TokenType.SEMICOLON, position=buffer.position(iterator.index)))

            elif cchar.isspace():
                if cchar == "\n":
                    if buffer.v:
                        iterator.real_index -= 1
                        tokens.append(buffer.tokenize(iterator.index))
                        iterator.real_index += 1

                    tokens.append(_Token(type=_TokenType.NEWLINE, position=buffer.position(iterator.index)))
                    buffer.reset(iterator.index)
                    buffer.line += 1

                elif cchar in [" ", "\t"]:
                    if buffer.v:
                        iterator.real_index -= 1
                        tokens.append(buffer.tokenize(iterator.index))
                        iterator.real_index += 1

                    tokens.append(_Token(type=_TokenType.WHITESPACE, position=buffer.position(iterator.index)))
                    buffer.reset(iterator.index)

            elif cchar in TDv:
                for TypeDefVar, TypeDef in zip(TDv, TD):
                    if TypeDefVar == cchar:
                        if buffer.v:
                            if cchar == ".":
                                buffer.v += "."
                                buffer.t = _TokenType.FILE
                                break
                            else:
                                iterator.real_index -= 1
                                tokens.append(buffer.tokenize(iterator.index))
                                iterator.real_index += 1

                        tokens.append(_Token(type=TypeDef, position=buffer.position(iterator.index)))
                        buffer.reset(iterator.index)
                        break

            else:
                if cchar is not None:
                    buffer.v += cchar
                    buffer.t = _TokenType.IDENTIFIER if buffer.t is None else buffer.t

                    if buffer.v.lower() in ["true", "false"] and not str(await iterator.peek()).isalpha():
                        buffer.t = _TokenType.BOOLEAN
                        tokens.append(buffer.tokenize(iterator.index))

            cchar = await iterator.advance()

        return tokens

    @staticmethod
    @xsync.as_hybrid()
    def __parse(tokens: List[_Token], safe: bool = True) -> Tuple[Dict[str, Any], Dict[str, _DictionaryTyping]]:
        # check if there is an eof token at the end, if not raise an exception
        eof = tokens[len(tokens) - 1]
        if eof.type != _TokenType.EOF:
            raise LonlException.Parser.ParsingException(
                "The given tokens do not contain an EndOfFile token and the "
                "token list is therefore not correctly lexed."
            )

        # cleaning the token list and removing TokenType WHITESPACE, NEWLINE, EoF
        tokens = [tok for tok in tokens if tok.type not in (_TokenType.WHITESPACE, _TokenType.NEWLINE, _TokenType.EOF)]

        # creating s list with TokenType
        TDv = [TDv.value for TDv in _TokenType]

        # creating 2 dicts, one for values one for typing
        value_dict: Dict[Any, Any] = {}
        typing_dict: Dict[Any, _DictionaryTyping] = {}

        # creating the iterator
        toks = _CustomIterator(tokens)

        # getting the first token
        ctok = toks.advance()

        while ctok:

            if ctok.type == _TokenType.IDENTIFIER:
                if toks.peek().type == _TokenType.LARROW and \
                        toks.peek(2).type == _TokenType.IDENTIFIER and \
                        toks.peek(3).type == _TokenType.RARROW and \
                        toks.peek(4).type == _TokenType.EQUALS:

                    temp1: _Token = ctok
                    # check if identifier already exists
                    if temp1.value in value_dict:
                        pos = Lonl.__exception_pos(temp1.position)
                        raise LonlException.Parser.ParsingException(
                            f"Failed to parse identifier '"
                            f"{temp1.value}' because another "
                            f"identifier with that name already "
                            f"exists. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )
                    temp_type = None

                    ctok = toks.advance(2)
                    temp2 = ctok

                    if temp2.value in TDv:
                        for dtok in TDv:
                            if dtok == temp2.value:
                                temp_type = dtok
                                break

                        ctok = toks.advance(3)
                        match temp_type:
                            case _TokenType.STRING.value:
                                value_dict[temp1.value] = ctok.value.replace("\"", "").replace("'", "")
                                typing_dict[temp1.value] = _DictionaryTyping(type=str)

                            case _TokenType.INTEGER.value:
                                value_dict[temp1.value] = int(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=int)

                            case _TokenType.FLOAT.value:
                                value_dict[temp1.value] = float(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=float)

                            case _TokenType.BOOLEAN.value:
                                if str(ctok.value).lower() == "true":
                                    value_dict[temp1.value] = True
                                elif str(ctok.value).lower() == "false":
                                    value_dict[temp1.value] = False
                                else:
                                    pos = Lonl.__exception_pos(ctok.position)
                                    raise LonlException.Parser.ParsingException(
                                        f"Failed to parse '"
                                        f"{ctok.value}' to type"
                                        f" '{temp2.value}'. Line: '{pos[2]}', region: '{pos[0]}' - "
                                        f"'{pos[1]}'."
                                    )
                                typing_dict[temp1.value] = _DictionaryTyping(type=bool)

                            case _TokenType.ENVIRONMENT.value:
                                if safe:
                                    raise LonlException.SafeModeEnabled(
                                        "Safe mode is enabled, therefore you cannot "
                                        "read environment variables."
                                    )
                                value_dict[temp1.value] = os.environ.get(ctok.value, None)
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.ENVIRONMENT)

                            case _TokenType.FILE.value:
                                with open("./" + str(ctok.value), "r") as file:
                                    value_dict[temp1.value] = file.read()
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.FILE)

                            case _TokenType.JSON.value:
                                value_dict[temp1.value] = js_load_str(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.JSON)

                            case _:
                                pos = Lonl.__exception_pos(temp1.position)
                                _, stop = _, _ = temp2.position
                                raise LonlException.Parser.ParsingException(
                                    f"Could not identify the type for identifier '{temp1.value}', type: '{temp_type}'."
                                    f"Line: '{pos[2]}', region: '{pos[0]}' - '{stop}'"
                                )

                        if toks.peek().type != _TokenType.SEMICOLON:
                            pos = Lonl.__exception_pos(ctok.position, 2)
                            raise LonlException.Parser.ParsingException(
                                f"Parser was not able to find a semicolon at "
                                f"the end of the variable '{ctok.value}'. "
                                f"Line '{pos[2]}', region: '{pos[0]}' - '"
                                f"{pos[1]}'"
                            )

                    else:
                        pos = Lonl.__exception_pos(ctok.position)
                        raise LonlException.Parser.ParsingException(
                            f"Identifier '{temp1.value}' has no valid type "
                            f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )

                elif toks.peek().type == _TokenType.LARROW and \
                        toks.peek(2).type == _TokenType.IDENTIFIER and \
                        toks.peek(3).type == _TokenType.LARROW and \
                        toks.peek(4).type == _TokenType.IDENTIFIER and \
                        toks.peek(5).type == _TokenType.RARROW and \
                        toks.peek(6).type == _TokenType.RARROW:

                    identifier = ctok
                    # check if identifier already exists
                    if identifier.value in value_dict:
                        pos = Lonl.__exception_pos(identifier.position)
                        raise LonlException.Parser.ParsingException(
                            f"Failed to parse identifier '"
                            f"{identifier.value}' because another "
                            f"identifier with that name already "
                            f"exists. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )
                    ctok = toks.advance(2)
                    temp1 = ctok
                    ctok = toks.advance(2)
                    temp2 = ctok
                    ctok = toks.advance(4)

                    temp_type1 = None
                    temp_type2 = None

                    if temp1.value in TDv:
                        for dtok in TDv:
                            if dtok == temp1.value:
                                temp_type1 = dtok
                                break

                        if temp2.value in TDv:
                            for dtok in TDv:
                                if dtok == temp2.value:
                                    temp_type2 = dtok
                                    break

                            if temp_type1 == _TokenType.LIST.value:

                                match temp_type2:
                                    case _TokenType.STRING.value:
                                        value_dict[identifier.value] = [x.replace("\"", "").replace("'", "") for x in
                                                                        Lonl.__str_to_list(ctok.value)]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=str)

                                    case _TokenType.INTEGER.value:
                                        value_dict[identifier.value] = [int(x) for x in Lonl.__str_to_list(ctok.value)]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=int)

                                    case _TokenType.FLOAT.value:
                                        value_dict[identifier.value] = [float(x) for x in Lonl.__str_to_list(
                                            ctok.value
                                        )]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=float)

                                    case _TokenType.BOOLEAN.value:
                                        new_list = []
                                        for obj in Lonl.__str_to_list(ctok.value):
                                            if obj.lower() == "true":
                                                new_list.append(True)
                                            elif obj.lower() == "false":
                                                new_list.append(False)
                                            else:
                                                pos = Lonl.__exception_pos(ctok.position)
                                                raise LonlException.Parser.ParsingException(
                                                    f"Failed to parse '"
                                                    f"{ctok.value}' to types"
                                                    f" '{temp1}', '{temp2}'. Line: '{pos[2]}', region: '{pos[0]}' - "
                                                    f"'{pos[1]}'."
                                                )
                                        value_dict[identifier.value] = new_list
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=bool)

                                    case _TokenType.ENVIRONMENT.value:
                                        if safe:
                                            raise LonlException.SafeModeEnabled(
                                                "Safe mode is enabled, therefore you cannot "
                                                "read environment variables."
                                            )

                                        new_list = []
                                        for obj in Lonl.__str_to_list(ctok.value):
                                            new_list.append(os.environ.get(obj.value, None))

                                        value_dict[temp1.value] = new_list
                                        typing_dict[temp1.value] = _DictionaryTyping(
                                            type=list,
                                            type2=str,
                                            tokentype=_TokenType.ENVIRONMENT
                                        )
                                    case _:
                                        start, _ = _, line = identifier.position
                                        _, stop = _, _ = temp2.position
                                        raise LonlException.Parser.ParsingException(
                                            f"Could not identify the type for identifier '{identifier.value}', "
                                            f"types: '{temp1}', '{temp2}'. Maybe type '{temp2}' is not supported by "
                                            f"type '{temp1}'?"
                                            f"Line: '{line}', region: '{start}' - '{stop}'"
                                        )

                            elif temp_type1 == _TokenType.FILE.value:
                                match temp_type2:
                                    case _TokenType.JSON.value:
                                        with open("./" + str(ctok.value), "r") as file:
                                            value_dict[temp1.value] = js_load(file)
                                        typing_dict[temp1.value] = _DictionaryTyping(
                                            type=str,
                                            tokentype=_TokenType.FILE,
                                            value=ctok.value
                                        )

                            else:
                                pos = Lonl.__exception_pos(temp1.position)
                                raise LonlException.Parser.ParsingException(
                                    f"Could not identify the type for identifier '{temp1.value}', type: '{temp1}'."
                                    f"Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                                )

                        else:
                            pos = Lonl.__exception_pos(temp2.position)
                            raise LonlException.Parser.ParsingException(
                                f"Identifier '{temp2.value}' has no valid type "
                                f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                            )

                    else:
                        pos = Lonl.__exception_pos(temp1.position)
                        raise LonlException.Parser.ParsingException(
                            f"Identifier '{temp1.value}' has no valid type "
                            f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )

            ctok = toks.advance()

        return value_dict, typing_dict

    @staticmethod
    @xsync.set_async_impl(__parse)
    async def __async_parse(tokens: List[_Token], safe: bool = True) \
            -> Tuple[Dict[str, Any], Dict[str, _DictionaryTyping]]:
        # check if there is an eof token at the end, if not raise an exception
        eof = tokens[len(tokens) - 1]
        if eof.type != _TokenType.EOF:
            raise LonlException.Parser.ParsingException(
                "The given tokens do not contain an EndOfFile token and the "
                "token list is therefore not correctly lexed."
            )

        # cleaning the token list and removing TokenType WHITESPACE, NEWLINE, EoF
        tokens = [tok for tok in tokens if tok.type not in (_TokenType.WHITESPACE, _TokenType.NEWLINE, _TokenType.EOF)]

        # creating s list with TokenType
        TDv = [TDv.value for TDv in _TokenType]

        # creating 2 dicts, one for values one for typing
        value_dict: Dict[Any, Any] = {}
        typing_dict: Dict[Any, _DictionaryTyping] = {}

        # creating the iterator
        toks = _CustomIterator(tokens)

        # getting the first token
        ctok = await toks.advance()

        while ctok:

            if ctok.type == _TokenType.IDENTIFIER:
                if (await toks.peek()).type == _TokenType.LARROW and \
                        (await toks.peek(2)).type == _TokenType.IDENTIFIER and \
                        (await toks.peek(3)).type == _TokenType.RARROW and \
                        (await toks.peek(4)).type == _TokenType.EQUALS:

                    temp1: _Token = ctok
                    # check if identifier already exists
                    if temp1.value in value_dict:
                        pos = Lonl.__exception_pos(temp1.position)
                        raise LonlException.Parser.ParsingException(
                            f"Failed to parse identifier '"
                            f"{temp1.value}' because another "
                            f"identifier with that name already "
                            f"exists. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )
                    temp_type = None

                    ctok = await toks.advance(2)
                    temp2 = ctok

                    if temp2.value in TDv:
                        for dtok in TDv:
                            if dtok == temp2.value:
                                temp_type = dtok
                                break

                        ctok = await toks.advance(3)
                        match temp_type:
                            case _TokenType.STRING.value:
                                value_dict[temp1.value] = ctok.value.replace("\"", "").replace("'", "")
                                typing_dict[temp1.value] = _DictionaryTyping(type=str)

                            case _TokenType.INTEGER.value:
                                value_dict[temp1.value] = int(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=int)

                            case _TokenType.FLOAT.value:
                                value_dict[temp1.value] = float(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=float)

                            case _TokenType.BOOLEAN.value:
                                if str(ctok.value).lower() == "true":
                                    value_dict[temp1.value] = True
                                elif str(ctok.value).lower() == "false":
                                    value_dict[temp1.value] = False
                                else:
                                    pos = Lonl.__exception_pos(ctok.position)
                                    raise LonlException.Parser.ParsingException(
                                        f"Failed to parse '"
                                        f"{ctok.value}' to type"
                                        f" '{temp2.value}'. Line: '{pos[2]}', region: '{pos[0]}' - "
                                        f"'{pos[1]}'."
                                    )
                                typing_dict[temp1.value] = _DictionaryTyping(type=bool)

                            case _TokenType.ENVIRONMENT.value:
                                if safe:
                                    raise LonlException.SafeModeEnabled(
                                        "Safe mode is enabled, therefore you cannot "
                                        "read environment variables."
                                    )
                                value_dict[temp1.value] = os.environ.get(ctok.value, None)
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.ENVIRONMENT)

                            case _TokenType.FILE.value:
                                with open("./" + str(ctok.value), "r") as file:
                                    value_dict[temp1.value] = file.read()
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.FILE)

                            case _TokenType.JSON.value:
                                value_dict[temp1.value] = js_load_str(ctok.value)
                                typing_dict[temp1.value] = _DictionaryTyping(type=str, tokentype=_TokenType.JSON)

                            case _:
                                pos = Lonl.__exception_pos(temp1.position)
                                _, stop = _, _ = temp2.position
                                raise LonlException.Parser.ParsingException(
                                    f"Could not identify the type for identifier '{temp1.value}', type: '{temp_type}'."
                                    f"Line: '{pos[2]}', region: '{pos[0]}' - '{stop}'"
                                )

                        if (await toks.peek()).type != _TokenType.SEMICOLON:
                            pos = Lonl.__exception_pos(ctok.position, 2)
                            raise LonlException.Parser.ParsingException(
                                f"Parser was not able to find a semicolon at "
                                f"the end of the variable '{ctok.value}'. "
                                f"Line '{pos[2]}', region: '{pos[0]}' - '"
                                f"{pos[1]}'"
                            )

                    else:
                        pos = Lonl.__exception_pos(ctok.position)
                        raise LonlException.Parser.ParsingException(
                            f"Identifier '{temp1.value}' has no valid type "
                            f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )

                elif (await toks.peek()).type == _TokenType.LARROW and \
                        (await toks.peek(2)).type == _TokenType.IDENTIFIER and \
                        (await toks.peek(3)).type == _TokenType.LARROW and \
                        (await toks.peek(4)).type == _TokenType.IDENTIFIER and \
                        (await toks.peek(5)).type == _TokenType.RARROW and \
                        (await toks.peek(6)).type == _TokenType.RARROW:

                    identifier = ctok
                    # check if identifier already exists
                    if identifier.value in value_dict:
                        pos = Lonl.__exception_pos(identifier.position)
                        raise LonlException.Parser.ParsingException(
                            f"Failed to parse identifier '"
                            f"{identifier.value}' because another "
                            f"identifier with that name already "
                            f"exists. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )
                    ctok = await toks.advance(2)
                    temp1 = ctok
                    ctok = await toks.advance(2)
                    temp2 = ctok
                    ctok = await toks.advance(4)

                    temp_type1 = None
                    temp_type2 = None

                    if temp1.value in TDv:
                        for dtok in TDv:
                            if dtok == temp1.value:
                                temp_type1 = dtok
                                break

                        if temp2.value in TDv:
                            for dtok in TDv:
                                if dtok == temp2.value:
                                    temp_type2 = dtok
                                    break

                            if temp_type1 == _TokenType.LIST.value:

                                match temp_type2:
                                    case _TokenType.STRING.value:
                                        value_dict[identifier.value] = [x.replace("\"", "").replace("'", "") for x in
                                                                        Lonl.__str_to_list(ctok.value)]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=str)

                                    case _TokenType.INTEGER.value:
                                        value_dict[identifier.value] = [int(x) for x in Lonl.__str_to_list(ctok.value)]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=int)

                                    case _TokenType.FLOAT.value:
                                        value_dict[identifier.value] = [float(x) for x in Lonl.__str_to_list(
                                            ctok.value
                                        )]
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=float)

                                    case _TokenType.BOOLEAN.value:
                                        new_list = []
                                        for obj in Lonl.__str_to_list(ctok.value):
                                            if obj.lower() == "true":
                                                new_list.append(True)
                                            elif obj.lower() == "false":
                                                new_list.append(False)
                                            else:
                                                pos = Lonl.__exception_pos(ctok.position)
                                                raise LonlException.Parser.ParsingException(
                                                    f"Failed to parse '"
                                                    f"{ctok.value}' to types"
                                                    f" '{temp1}', '{temp2}'. Line: '{pos[2]}', region: '{pos[0]}' - "
                                                    f"'{pos[1]}'."
                                                )
                                        value_dict[identifier.value] = new_list
                                        typing_dict[identifier.value] = _DictionaryTyping(type=list, type2=bool)

                                    case _TokenType.ENVIRONMENT.value:
                                        if safe:
                                            raise LonlException.SafeModeEnabled(
                                                "Safe mode is enabled, therefore you cannot "
                                                "read environment variables."
                                            )

                                        new_list = []
                                        for obj in Lonl.__str_to_list(ctok.value):
                                            new_list.append(os.environ.get(obj.value, None))

                                        value_dict[temp1.value] = new_list
                                        typing_dict[temp1.value] = _DictionaryTyping(
                                            type=list,
                                            type2=str,
                                            tokentype=_TokenType.ENVIRONMENT
                                        )
                                    case _:
                                        start, _ = _, line = identifier.position
                                        _, stop = _, _ = temp2.position
                                        raise LonlException.Parser.ParsingException(
                                            f"Could not identify the type for identifier '{identifier.value}', "
                                            f"types: '{temp1}', '{temp2}'. Maybe type '{temp2}' is not supported by "
                                            f"type '{temp1}'?"
                                            f"Line: '{line}', region: '{start}' - '{stop}'"
                                        )

                            elif temp_type1 == _TokenType.FILE.value:
                                match temp_type2:
                                    case _TokenType.JSON.value:
                                        with open("./" + str(ctok.value), "r") as file:
                                            value_dict[temp1.value] = js_load(file)
                                        typing_dict[temp1.value] = _DictionaryTyping(
                                            type=str,
                                            tokentype=_TokenType.FILE,
                                            value=ctok.value
                                        )

                            else:
                                pos = Lonl.__exception_pos(temp1.position)
                                raise LonlException.Parser.ParsingException(
                                    f"Could not identify the type for identifier '{temp1.value}', type: '{temp1}'."
                                    f"Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                                )

                        else:
                            pos = Lonl.__exception_pos(temp2.position)
                            raise LonlException.Parser.ParsingException(
                                f"Identifier '{temp2.value}' has no valid type "
                                f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                            )

                    else:
                        pos = Lonl.__exception_pos(temp1.position)
                        raise LonlException.Parser.ParsingException(
                            f"Identifier '{temp1.value}' has no valid type "
                            f"definition. Line: '{pos[2]}', region: '{pos[0]}' - '{pos[1]}'"
                        )

            ctok = await toks.advance()

        return value_dict, typing_dict

    @staticmethod
    @xsync.as_hybrid()
    def read(filestream: TextIO, safe=True) -> LonlDictionary:
        """
        Method for reading a filestream and returns a LonlDictionary (Can be used async and sync)
        :param filestream: filestream for reading a `.lonl` file
        :type filestream: TextIO
        :param safe: Forbids the use for methods that could harm the system and the program.
        (Only disable on trusted systems and when using trusted `.lonl` files
        :type safe: bool
        :return: lonlDictionary
        """
        contents = filestream.read()

        tokens: List[_Token] = Lonl.__lex(contents)
        tokens.append(_Token(type=_TokenType.EOF, value=len(tokens)))

        parsed = Lonl.__parse(tokens, safe)
        return LonlDictionary(*parsed)

    @staticmethod
    @xsync.set_async_impl(read)
    async def async_read(filestream: IO, safe=True) -> LonlDictionary:
        """
        Method for reading a filestream and returns a LonlDictionary (Can be used async and sync)
        :param filestream: filestream for reading a `.lonl` file
        :type filestream: TextIO
        :param safe: Forbids the use for methods that could harm the system and the program.
        (Only disable on trusted systems and when using trusted `.lonl` files
        :type safe: bool
        :return: lonlDictionary
        """
        contents = await filestream.read()

        tokens: List[_Token] = await Lonl.__lex(contents)
        tokens.append(_Token(type=_TokenType.EOF, value=len(tokens)))

        parsed = await Lonl.__parse(tokens, safe)
        return LonlDictionary(*parsed)


if __name__ == '__main__':
    with open("../Test.lonl", "r") as f:
        dictionary = Lonl.read(f, safe=False)

        print(dictionary)

        # tabulate
        # from tabulate import tabulate

        # table_list = [[tok.type, tok.value, tok.position[0][0], tok.position[0][1], tok.position[1]] for tok in
        # tokens_]

        # print(tabulate(table_list, headers=["Token Type", "Token Value", "Start", "End", "Line"]))
