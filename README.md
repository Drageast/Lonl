# lonl -> Luca's object notation language

## About 
As you can make out from the title, this is my custom object notation language for python 3.10+.
I created it because in languages like json, I was missing features like declaring an env variable
and getting its value at need. 

I even have one use case example. If you write a Discord bot and want to publish your configuration 
file but do not want to leak your bot token, just declare that lonl finds the token in another file 
and done.

## Syntax

The syntax is fairly simple. 
```
Identifier<type> = Value;

Examples:

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

## Caution!

Please use this project with caution because it can sometimes be a really buggy mess, and I 
still have to work on a way to write the ConlDictionary - Or any Dictionary at all back into 
that file format.

So, please wait **until this project gets released on pypi**!

## Contribution / Request 

If anyone wants to contribute, please contact me via my e-mail address on my profile, 
goes for any request for lonl.
