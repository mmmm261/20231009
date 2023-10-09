# 元类会自动获取通常传给`type`的参数
def upper_attr(_class, _object, _attr):
    """
      返回一个类对象，将其属性置为大写
    """

    # 过滤出所有开头不为'__'的属性，置为大写
    uppercase_attr = {}
    for name, val in _attr.items():
        if not name.startswith('__'):
            uppercase_attr[name.upper()] = val
        else:
            uppercase_attr[name] = val

    # 利用'type'创建类，同时将这个类进行返回
    return type(_class, _object, uppercase_attr)


class Foo(metaclass=upper_attr):  # 创建对象时，其会经过 metaclass 来创建，再使用自身的方法进行实例化
    bar = 'bip'


if __name__ == '__main__':
    print(hasattr(Foo, 'bar'))
    print(hasattr(Foo, 'BAR'))

    f = Foo()
    print(f.BAR)


