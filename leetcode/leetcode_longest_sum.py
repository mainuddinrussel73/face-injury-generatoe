import math
from ftplib import print_line
from time import sleep

import numpy as np


class ListNode:
    def __init__(self, data):
        self.data = data  # Assign data to the node
        self.next = None  # Initialize the next node as None


class Solution:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def append(self, data):
        new_node = ListNode(data)
        if not self.head:
            self.head = new_node
        else:
            last = self.head
            while last.next:
                last = last.next
            last.next = new_node

        self.length += 1

    def display(self):
        current = self.head
        list = np.array([])
        while current:
            print(current.data, end='->')
            current = current.next
        print("]")




    def loop_sum(self, other):

        biggest =  0

        if self.length >= other.length:
            biggest = self.length
        else:
            biggest = other.length


        if  self.length != other.length:
            if self.length == biggest :

                for i in range(biggest - other.length):
                    other.append(0)
            elif other.length == biggest:

                for i in range(biggest - self.length):
                    self.append(0)




        sum = 0
        carry = 0
        quo = 0
        linked_list_t = Solution()

        for i in range(biggest):

            if i == biggest-1:
                temp_sum = carry + other.head.data + self.head.data

                if temp_sum >= 10:
                    carry = math.floor(temp_sum / 10)
                    quo = temp_sum % 10
                    linked_list_t.append(quo)
                    linked_list_t.append(carry)

                else:
                    linked_list_t.append(temp_sum)
            elif i < biggest:
                temp_sum = carry + other.head.data + self.head.data
                self.head = self.head.next
                other.head = other.head.next
                if temp_sum >= 10:
                    carry = math.floor(temp_sum / 10)
                    quo = temp_sum % 10
                    linked_list_t.append(quo)


                else:
                    linked_list_t.append(temp_sum)


        linked_list_t.display()


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy_head = ListNode()
        dummy = dummy_head
        carry = 0
        quo = 0
        temp_sum = 0



        while l1  or l2:

            x = l1.val if l1 else 0
            y = l2.val if l2 else 0

            temp_sum = carry + x + y
            print(temp_sum)

            quo = temp_sum % 10
            carry = (temp_sum // 10)


            if temp_sum >= 10:
                curr = (ListNode(quo))
                dummy.next = curr
                dummy = dummy.next

            else:
                curr = (ListNode(temp_sum))
                dummy.next = curr
                dummy = dummy.next



            if l1: l1 = l1.next
            if l2: l2 = l2.next

        if temp_sum >= 10:
            curr = (ListNode(carry))
            dummy.next = curr


        return dummy_head.next  # Return the next node from the dummy head, which is the actual result


def create_linked_list(digits):
    dummy_head = ListNode(0)
    current = dummy_head
    for digit in digits:
        current.next = ListNode(digit)
        current = current.next
    return dummy_head.next

# Helper function to print a linked list
def print_linked_list(node):
    while node:
        print(node.val, end=" -> " if node.next else "")
        node = node.next
    print()

# Input the digits for the two numbers
digits1 = list(map(int, input("Enter the digits for the first number in reverse order (space-separated): ").split()))
digits2 = list(map(int, input("Enter the digits for the second number in reverse order (space-separated): ").split()))

# Create two linked lists from the input digits
l1 = create_linked_list(digits1)
l2 = create_linked_list(digits2)

# Perform addition using the Solution class
solution = Solution()
result = solution.addTwoNumbers(l1, l2)

# Output the result as a linked list
print("The sum as a linked list:")
print_linked_list(result)




