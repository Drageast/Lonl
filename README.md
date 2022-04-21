# LONL -> Luca's object notation language

## About 
As you can make out from the title, this is my custom object notation language for python 3.10+.
I created it because in languages like json, I was missing features like declaring an env variable
and getting its value as needed. 

I even have one use case example. If you write a Discord bot and want to publish your configuration 
file but do not want to leak your bot token, just declare it so that `lonl` finds the token in another file 
and done.

## Syntax

The syntax is fairly simple. 
```
Identifier<type> = Value;

# Examples: 


# everything after a hashtag (up to a newline) counts as comment

String<str> = "Hello, world!";
Integer<int> = 19;
Float<float> = 3.99;
File<file> = password.txt;
Environment<env> = my_env;
Json<json> = {"Hello": "World!", "Fruits": {"Apple": "Fruit"}, "Lists": ["1", "2"]};
```
Some types even support further type annotations. 
```
Lists<list<str>> = ["Hello", "World!"];
FileAsJson<file<json>> = secret_config.json;
```

Lists support everything except for `json` and `files`.

## How to use

### Load `.lonl` files and string

```python
# Import Lonl from Lonl
from Lonl import Lonl


# Loading lonl from a file:
def with_file():
    with open("Test.lonl", "r") as f:
        data = Lonl.load(f, safe=True)
        
        print(data)

        
# loading from a string:
def with_string():
    data = Lonl.loads("TestVar<int> = 19;", safe=True)

    print(data)
```

 ### Dump `.lonl` files and strings

```python
# Import Lonl from Lonl
from Lonl import Lonl

# you need a LonlDictionary or a normal dictionary to pass in
# the normal dictionary will be converted to a LonlDictionary

data = {"Hello": "World!", "Nested": {"Fruits": ["Apples", "Cherry's"]}}

# dumping lonl in a file:
def with_file():
    with open("Test.lonl", "w") as f:
        Lonl.dump(f, data)

        
# dumping to a string:
def with_string():
    lonl_string = Lonl.dumps(data)

    print(lonl_string)
```

## Caution!

Please use this project with caution because it can sometimes be a really buggy mess, and I 
still have to work on many things and rewrite a lot as well.

So, please wait **until this project gets released on pypi**!

## Contribution / Request 

If anyone wants to contribute, please contact me via my e-mail address on my profile, 
same goes for any feature requests for lonl.
