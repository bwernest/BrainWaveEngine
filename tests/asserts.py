"""___Functions_____________________________________________________________"""

def assertEqual(arg1: any, arg2: any) -> None:
    assert arg1 == arg2, f"Arguments inégaux ! {arg1} / {arg2}"

def assertNotEqual(arg1: any, arg2: any) -> None:
    assert arg1 != arg2, f"Arguments égaux ! {arg1} / {arg2}"

def assertIsInstance(arg1: any, _type: type) -> None:
    assert isinstance(arg1, _type), f"Argument est de type {type(arg1)} et non de type {_type}"

def assertListIsInstance(arg1: any, _type: type) -> None:
    for elem in arg1:
        assertIsInstance(elem, _type)

def assertListEqual(arg1: any, arg2: any) -> None:
    for elem1, elem2 in zip(arg1, arg2):
        assertEqual(elem1, elem2)
