from ftplib import print_line

from scipy.signal import max_len_seq


class Solution:
    def __init__(self):
        self.map_string = {}

    def is_palindrome(self,s):
        cleaned_str = ''.join(char.lower() for char in s if char.isalnum())

        return cleaned_str == cleaned_str[::-1]

    def find_all_substrings_between(self,s, char):
        positions = [i for i, c in enumerate(s) if c == char]

        if len(positions) < 2:
            return char

        # Collect substrings between every pair of occurrences
        max_sub = ""
        max_len = 0

        for i in range(len(positions) -1 ):  # Only need to go till the second-to-last position
            for j in range(i + 1 , len(positions) ):  # Ensure j > i to avoid self-pairing
                x = s[positions[i]:positions[j] + 1]
                if self.is_palindrome(x):

                    if len(x) >= max_len:
                        max_len = max(max_len, len(x))
                        max_sub = x

        return max_sub

    def palindrome_find(self, s):
        max_len_seq = ""
        max_len = 0
        for i in range(len(s)):
            curr = s[i]
            if curr not in self.map_string:
                temp_str = s
                temp_subs = self.find_all_substrings_between(temp_str ,curr)
                if temp_subs:
                    if len(temp_subs) > max_len:
                        max_len = len(temp_subs)
                        max_len_seq = temp_subs

                self.map_string[curr] = i


        return max_len_seq















print(Solution().palindrome_find("jkexvzsqshsxyytjmmhauoyrbxlgvdovlhzivkeixnoboqlfemfzytbolixqzwkfvnpacemgpotjtqokrqtnwjpjdiidduxdprngvitnzgyjgreyjmijmfbwsowbxtqkfeasjnujnrzlxmlcmmbdbgryknraasfgusapjcootlklirtilujjbatpazeihmhaprdxoucjkynqxbggruleopvdrukicpuleumbrgofpsmwopvhdbkkfncnvqamttwyvezqzswmwyhsontvioaakowannmgwjwpehcbtlzmntbmbkkxsrtzvfeggkzisxqkzmwjtbfjjxndmsjpdgimpznzojwfivgjdymtffmwtvzzkmeclquqnzngazmcfvbqfyudpyxlbvbcgyyweaakchxggflbgjplcftssmkssfinffnifsskmsstfclpjgblfggxhckaaewyygcbvblxypduyfqbvfcmzagnznquqlcemkzzvtwmfftmydjgvifwjoznzpmigdpjsmdnxjjfbtjwmzkqxsizkggefvztrsxkkbmbtnmzltbchepwjwgmnnawokaaoivtnoshywmwszqzevywttmaqvncnfkkbdhvpowmspfogrbmuelupcikurdvpoelurggbxqnykjcuoxdrpahmhiezaptabjjulitrilkltoocjpasugfsaarnkyrgbdbmmclmxlzrnjunjsaefkqtxbwoswbfmjimjyergjygzntivgnrpdxuddiidjpjwntqrkoqtjtopgmecapnvfkwzqxilobtyzfmeflqobonxiekvizhlvodvglxbryouahmmjtyyxshsqszvxekj"))