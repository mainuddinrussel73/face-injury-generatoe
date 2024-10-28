def two_sum(nums, target):
    # Dictionary to store numbers and their indices
    seen = {}

    # Iterate through the array
    for i, num in enumerate(nums):
        # Calculate the complement needed to reach the target
        complement = target - num

        # If the complement is already in the dictionary, return the indices
        if complement in seen:
            return [seen[complement], i]

        # Otherwise, store the current number with its index
        seen[num] = i


nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)