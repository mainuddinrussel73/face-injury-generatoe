

class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("Dequeue from empty queue")

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Peek from empty queue")

    def size(self):
        return len(self.items)

    def find(self, item):
        return item in self.items

    def find_index(self, item):
        try:
            return self.items.index(item)
        except ValueError:
            return -1  # Item not found

    def clear(self):
        self.items = []

    def __str__(self):
        return str(self.items)

    def remove_leftward(self, index):
        if index < 0 or index >= len(self.items):
            raise IndexError("Index out of range")
        # Remove all items up to and including the specified index
        self.items = self.items[index + 1:]

class Solution:

    def lengthOfLongestSubstring(self, s: str) -> int:
        q = Queue()
        my_dict = {}
        leng = 0

        for i in range(len(s)):
            if q.find(s[i]):
                idx = q.find_index(s[i])
                leng = q.size()
                my_dict[leng] = q.__str__()
                q.remove_leftward(idx)
                q.enqueue(s[i])
                leng = q.size()
            else:
                q.enqueue(s[i])
                leng = q.size()

        if len(s) > 0:
            my_dict[leng] = q.__str__()
        q.clear()
        leng = 0

        print(my_dict)

        if len(my_dict) == 0:
            return 0
        else:
            max_key = max(my_dict)
            return max_key


s = Solution()


print("Solution" + str(s.lengthOfLongestSubstring("abcdefcklml opqrstuv fgj ")))


