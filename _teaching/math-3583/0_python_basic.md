---
layout: page
permalink: /teaching/math-3583/0_python_basic/
title: Python Basic
---

### **Installing Anaconda and Launching Jupyter Notebook**

#### **Introduction**
Anaconda is a popular distribution of Python, which comes with many scientific libraries and tools, including Jupyter Notebook – an interactive computing environment that allows users to write and run code in a web browser.

#### **Installation on Mac:**

1. **Download Installer:**
   - Visit the [Anaconda Downloads page](https://www.anaconda.com/products/distribution).
   - Select the Python 3.x version's MacOS installer.

2. **Install:**
   - Open the downloaded file and follow the installation instructions.
   
3. **Verify Installation:**
   - Open your terminal.
   - Type `conda list`. If Anaconda was installed correctly, you'll see a list of installed packages.

4. **Launch Jupyter Notebook:**
   - In the terminal, type `jupyter notebook`.
   - Your web browser will open with the Jupyter Notebook interface.

#### **Installation on Windows:**

1. **Download Installer:**
   - Visit the [Anaconda Downloads page](https://www.anaconda.com/products/distribution).
   - Select the Python 3.x version's Windows installer.

2. **Install:**
   - Open the downloaded .exe file.
   - Follow the installation instructions, ensuring you choose "Add Anaconda to PATH" during installation.

3. **Verify Installation:**
   - Search for and open the "Anaconda Prompt" from the Start menu.
   - Type `conda list`. If Anaconda was installed correctly, you'll see a list of installed packages.

4. **Launch Jupyter Notebook:**
   - In the Anaconda Prompt, type `jupyter notebook`.
   - Your web browser will open with the Jupyter Notebook interface.

#### **Using Jupyter Notebook:**

1. **New Python Notebook:**
   - Click on the "New" button and select "Python 3" to start a new Python notebook.

2. **Running Code:**
   - Write Python code in the cell.
   - Press `Shift + Enter` to run the code in the cell.

3. **Saving Work:**
   - Click on the floppy disk icon or use `Cmd + S` (Mac) or `Ctrl + S` (Windows) to save your notebook.

### Python Basics

Python is an interpreted, object-oriented, high-level programming language known for its dynamic semantics. Its elegance and simplicity, combined with its power, make it a popular choice for both beginners and seasoned developers.

**Interpreted vs. Compiled Language**:
- Interpreted languages, like Python, are executed line-by-line directly by the language's interpreter. This means you can run code immediately but might see slower execution times.
- Compiled languages convert code into machine-level instructions before execution. They typically offer faster execution but lack the immediacy of interpreted languages.

**Object-oriented vs. Procedural Programming**:
- Object-oriented programming (OOP) structures code around objects, allowing for encapsulation, inheritance, and polymorphism. Python supports OOP paradigms.
- Procedural programming, in contrast, focuses on functions and procedures to perform tasks.

**High-level vs. Low-level Language**:
- High-level languages, like Python, abstract away most of the complex, machine-specific details, allowing developers to write code that's readable and maintainable.
- Low-level languages are closer to machine code and require more detail, offering speed and optimization at the expense of ease of use.

**Python's Key Features**:
1. **Easy to Learn and Use**: Python's syntax is designed to be readable and straightforward, making it especially friendly for newcomers.
2. **Expressive Language**: Python lets you achieve more with fewer lines of code.
3. **Interpreted Language**: No need for a separate compilation step; code runs directly.
4. **Cross-platform Language**: Python runs on various platforms, from Windows and macOS to Linux and beyond.
5. **Free and Open Source**: Python comes with an OSI-approved open-source license.
6. **Object-oriented Language**: Enables modeling and structuring of code in an object-centric manner.
7. **Extensible and Integrated**: Easily integrate Python with other languages like C/C++, JAVA, Fortran, etc.
8. **Large Standard Library**: Python's extensive library offers modules for various functionalities, reducing the need to write every piece of code from scratch.

#### Python History
Python was conceived in the late 1980s by Guido van Rossum at the Centrum Wiskunde & Informatica (CWI) in the Netherlands. The first release, Python 1.0, was in 1994. Python 2.0, introduced in 2000, brought significant improvements, while Python 3.0, released in 2008, was a major overhaul that emphasized eliminating redundant structures and improving language consistency.

#### **1. Variable and Data Types**



```python
# Integer
x = 5
print(type(x))

# Float
y = 5.0
print(type(y))

# String
name = "John"
print(type(name))

# Boolean
flag = True
print(type(flag))
```

    <class 'int'>
    <class 'float'>
    <class 'str'>
    <class 'bool'>


#### 2. Calculations, Relational Operators


```python
a = 7
b = 3

# Basic calculations
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a % b)

# Relational operators
print(a > b)
print(a < b)
print(a == b)
print(a != b)
print(a >= b)
print(a <= b)
```

    10
    4
    21
    2.3333333333333335
    1
    True
    False
    False
    True
    True
    False


#### 3. Lists, List Index, len, max, min, sum


```python
lst = [5, 2, 8, 6, 1]

# Accessing elements by index
print(lst[0])  # First element
print(lst[-1])  # Last element

# List functions
print(len(lst))
print(max(lst))
print(min(lst))
print(sum(lst))
```

    5
    1
    5
    8
    1
    22


#### 4. Numpy Array


```python
import numpy as np

# Creating an array
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Basic operations
print(arr + 5)
print(arr * 2)
print(np.mean(arr))
```

    [1 2 3 4 5]
    [ 6  7  8  9 10]
    [ 2  4  6  8 10]
    3.0


5. Functions, if-statement


```python
def greet(name):
    if name == "Alice":
        return "Hello, Alice!"
    else:
        return f"Hello, {name}!"

print(greet("Bob"))
print(greet("Alice"))
```

    Hello, Bob!
    Hello, Alice!


#### 6. Dictionaries


```python
# Defining a dictionary
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Accessing values
print(person["name"])
print(person["age"])

# Adding a new key-value pair
person["job"] = "Engineer"
print(person)

# Getting all keys and values
print(person.keys())
print(person.values())
```

    John
    30
    {'name': 'John', 'age': 30, 'city': 'New York', 'job': 'Engineer'}
    dict_keys(['name', 'age', 'city', 'job'])
    dict_values(['John', 30, 'New York', 'Engineer'])
