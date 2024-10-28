import math
from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:

        a = len(nums1)
        b = len(nums2)

        if a <= b:
            arr_s = nums1
            arr_l = nums2
        else:
            arr_s = nums2
            arr_l = nums1
            temp = a
            a = b
            b = temp

        x = a
        y = b

        low = 0
        high = x

        while low <= high:

            position_x = ((low + high) // 2)
            position_y = ((x + y + 1) // 2) - position_x

            maxLeft_x = float("-inf") if position_x == 0 else arr_s[position_x - 1]
            minRight_x = float("inf") if position_x == x else arr_s[position_x]
            maxLeft_y = float("-inf") if position_y == 0 else arr_l[position_y - 1]
            minRight_y = float("inf") if position_y == y else arr_l[position_y]


            if maxLeft_x <= minRight_y and maxLeft_y <= minRight_x:

                if (x + y) % 2 == 0:

                    return float((max(maxLeft_x, maxLeft_y) + min(minRight_x, minRight_y)) / 2)
                else:
                    return float(max(maxLeft_x, maxLeft_y))
            elif maxLeft_x > minRight_y:
                high = position_x - 1

            else:
                low = position_x + 1


# Example usage
solution = Solution()
median = solution.findMedianSortedArrays([2], [])
print(median)  # Output: 2.0
