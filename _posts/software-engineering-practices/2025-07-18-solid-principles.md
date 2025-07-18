---
layout: post
title: "Understanding SOLID Principles in Python"
date: 2025-07-18 10:00:00 +0530
categories: [Software Engineering Practices]
tags: [SOLID, Software Design, Clean Code, Python]
---

The SOLID principles are foundational design guidelines every software engineer should know. They improve code readability, testability, and maintainability. In this post, we’ll cover:

1. Definition of each principle  
2. Python code samples  
3. Edge cases / where it breaks  
4. Follow-ups and important considerations  
5. Favorite interview questions  

---

## 1. Single Responsibility Principle (SRP)

**Definition**  
A class should have only one reason to change.

**Example**  
```python
class Invoice:
    def __init__(self, items):
        self.items = items

    def calculate_total(self):
        return sum(item['price'] for item in self.items)

class InvoicePrinter:
    def print_invoice(self, invoice):
        print(f"Total: {invoice.calculate_total()}")
```

### Where It Breaks
Combining calculation logic and printing in the same class.

### Follow-ups / Key Considerations

What constitutes “one reason to change” in your domain?

How do you decompose responsibilities in microservices?

#### Interview Question
Describe a class you refactored to adhere to SRP and the benefits you observed.

## 2. Open/Closed Principle (OCP)

**Definition**  
Software entities should be open for extension but closed for modification.

**Example**  
```python
class DiscountStrategy:
    def apply_discount(self, total):
        return total

class ChristmasDiscount(DiscountStrategy):
    def apply_discount(self, total):
        return total * 0.9
```

### Where It Breaks
Adding conditional logic for every new discount in a monolithic function or class.

### Follow-ups / Key Considerations
How would you apply new behaviors without touching existing code?

Can you illustrate this with plugin architectures or feature flags?

#### Interview Question
Explain how you would design a pricing engine to support multiple discount types without modifying core logic.

## 3. Liskov Substitution Principle (LSP)
**Definition**
Subtypes should be substitutable for their base types without altering program correctness.

**Example**
```python
class Bird:
    def fly(self):
        print("Flying")

class Sparrow(Bird):
    pass
```
### Where It Breaks
If you subclass Bird with Penguin and its fly() method raises an exception.

### Follow-ups / Key Considerations
How do you design class hierarchies to avoid violating LSP?

When is composition preferable to inheritance?

#### Interview Question
Can you describe a scenario where inheritance violated LSP and how you refactored it?

## 4. Interface Segregation Principle (ISP)
**Definition**
Clients should not be forced to depend on interfaces they do not use.

**Example**
```python
class Printer:
    def print(self):
        pass

class Scanner:
    def scan(self):
        pass

class MultiFunctionDevice(Printer, Scanner):
    def print(self):
        # printing logic
        pass

    def scan(self):
        # scanning logic
        pass
```

### Where It Breaks
A single interface that includes print, scan, fax, etc., forcing clients to implement unused methods.

### Follow-ups / Key Considerations
How do you structure interfaces (or abstract base classes) in Python to avoid this?

What role do mixins play here?

Interview Question
How would you refactor a monolithic interface into more focused abstractions?

## 5. Dependency Inversion Principle (DIP)
**Definition**
High-level modules should not depend on low-level modules; both should depend on abstractions.

**Example**
```python
from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def fetch(self):
        pass

class APIDataFetcher(DataFetcher):
    def fetch(self):
        return "Data from API"

class ReportGenerator:
    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def generate(self):
        print(f"Report: {self.fetcher.fetch()}")
```

### Where It Breaks
Hard-coding a specific data source inside ReportGenerator instead of depending on an abstraction.

### Follow-ups / Key Considerations

How do you apply DIP in Python without formal interfaces?

What patterns (e.g., factory, dependency injection frameworks) can help?

#### Interview Question
Describe how you would design a module to allow swapping data sources at runtime.

#### Interview-Favorite Questions
Which SOLID principle is hardest to apply in large microservices architectures?

How does the Open/Closed Principle relate to feature toggles?

Can you demonstrate the Dependency Inversion Principle in pure Python?

Describe a situation where you violated SRP and how you refactored the code.

What are the trade-offs between inheritance and composition regarding LSP?

