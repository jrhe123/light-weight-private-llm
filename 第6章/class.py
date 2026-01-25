# Python中类的示例

# 定义一个类
class Person:
    # 定义一个类属性
    name = ""
    age = 0

    # 定义一个构造方法
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 定义一个方法
    def say_hello(self):
        print("Hello, 我是%s, 今年%d岁" % (self.name, self.age))

class Student(Person):
    # 定义一个类属性
    school = ""

    # 定义一个构造方法
    def __init__(self, name, age, school):
        self.name = name
        self.age = age
        self.school = school

    # 定义一个方法
    def say_hello(self):
        print("Hello, 我是%s, 今年%d岁, 在%s上学" % (self.name, self.age, self.school))

# 创建一个对象
p = Person("张三", 20)
p.say_hello()

p.name = "李四"
p.age = 30
p.say_hello()

s = Student("王五", 25, "清华大学")
s.say_hello()
