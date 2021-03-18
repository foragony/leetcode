# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 Double Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

class Solution(object):
    def findLengthOfLCIS(self, nums):
        """
        674最长连续递增序列
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length < 1:
            return 0
        count = 1
        countmax = 1
        for i in range(length):
            if nums[i] < nums[i + 1]:
                count += 1
            else:
                if count > countmax:
                    countmax = count
                count += 1
        if count > countmax:
            countmax = count
        return countmax

    def runningSum(self, nums):
        """
        1480一维数组动态和
        :type nums: List[int]
        :rtype: List[int]
        """
        return_nums = [nums[0]]
        for i in range(len(nums)-1):
            return_nums.append(return_nums[-1]+nums[i+1])
        return return_nums

    def twoSum(self, nums, target):
        """
        1两数之和
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_len = len(nums)
        for i in range(nums_len-1):
            for j in range(nums_len-i-1):
                if (nums[i]+nums[j+i+1]) == target:
                    result = [i, i+j+1]
                    return result
        return None

    def reverse(self, x):
        """
        7整数反转
        :type x: int
        :rtype: int
        """
        i = 0
        y = 0
        flag = 0
        num = []
        if x < 0:
            flag = 1
            x = -x
        while x//10 > 0:
            num.append(x % 10)
            x = x//10
        num.append(x % 10)
        for a in num:
            y += a * 10**(len(num)-i-1)
            i += 1
        if flag == 1:
            y = -y
        if y > 2**31 - 1 or y < -(2**31):
            return 0
        return y

    def isPalindrome(self, x):
        """
        9回文数
        :type x: int
        :rtype: bool
        """
        y = x
        res = 0
        if x < 0:
            return False
        if x == 0:
            return True
        while x:
            res = res * 10 + x % 10
            x //= 10
        return y == res

    def romanToInt(self, s):
        """
        13罗马数字转整数
        :type s: str
        :rtype: int
        """
        a = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        y = 0
        for i in range(len(s)):
            if i + 1 < len(s) and a[s[i]] < a[s[i+1]]:
                y -= a[s[i]]
            else:
                y += a[s[i]]
        return y

    def longestCommonPrefix(self, strs):
        """
        14最长公共前缀
        :type strs: List[str]
        :rtype: str
        """
        if not strs: return ""
        s1 = min(strs)
        s2 = max(strs)
        for i,x in enumerate(s1):
            if x != s2[i]:
                return s2[:i]
        return s1

    def isValid(self, s):
        """
        20有效的括号
        :type s: str
        :rtype: bool
        """
        if len(s) < 1:
            return True
        aa = []
        for x in s:
            if x == '(' or x == '[' or x == '{':
                aa.append(x)
            else:
                if aa:
                    if ord(x) - ord(aa[-1]) == 1 or ord(x) - ord(aa[-1]) == 2:
                        aa.pop()
                    else:
                        return False
                else:
                    return False
        return not aa

    def mergeTwoLists(self, l1, l2):
        """
        21合并两个有序链表
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = ListNode()
        r = head
        while l1 and l2:
            if l1.val < l2.val:
                r.next = l1
                r = r.next
                l1 = l1.next
            else:
                r.next = l2
                r = r.next
                l2 = l2.next
        if l1:
            r.next = l1
        if l2:
            r.next = l2
        return head.next

    def removeDuplicates(self, nums):
        """
        26删除排序数组中的重复项
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        p = 0
        q = 1
        while q < len(nums):
            if nums[p] == nums[q]:
                q += 1
            else:
                p += 1
                nums[p] = nums[q]
                q += 1
        return p + 1

    def removeElement(self, nums, val):
        """
        27移除元素
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        p, q = 0, 0
        i = len(nums)
        while q < i:
            if nums[q] != val:
                nums[p] = nums[q]
                p += 1
            q += 1
        return p

    def strStr(self, haystack, needle):
        """
        28实现strStr()
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        h,n = len(needle),len(haystack)
        for i in range(n - h + 1):
            if haystack[i:h+i] == needle:
                return i
        else:
            return -1

    def searchInsert(self, nums, target):
        """
        35搜索插入位置
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lenth = len(nums)
        left = 0
        right = lenth - 1
        while right > left or right == left:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def maxSubArray(self, nums):
        """
        53最大子序和
        :type nums: List[int]
        :rtype: int
        """
        result = None
        fi = 0
        for i in range(len(nums)):
            fi = max(fi + nums[i], nums[i])
            if fi > result or result == None:
                result = fi
        return result

    def lengthOfLastWord(self, s):
        """
        58最后一个单词的长度
        :type s: str
        :rtype: int
        """
        length = len(s) - 1
        while s[length] == ' ' and length > -1:
            length -= 1
        if length < 0:
            return 0
        i = length
        while i > -1:
            if s[i] == ' ':
                break
            i -= 1
        return length - i

    def plusOne(self, digits):
        """
        66加一
        :type digits: List[int]
        :rtype: List[int]
        """
        length = len(digits) - 1
        for i in range(length + 1):
            if digits[length - i] == 9:
                digits[length - i] = 0
            else:
                digits[length - i] += 1
                return digits
        digits.insert(0, 1)
        return digits

    def addBinary(self, a, b):
        """
        67二进制求和
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a, 2)+int(b, 2))[2:]

    def mySqrt(self, x):
        """
        69x的平方根
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        i = 1
        while i**2 <= x:
            i += 1
        return i - 1

    def climbStairs(self, n):
        """
        70爬楼梯
        :type n: int
        :rtype: int
        """
        if n < 3:
            return n
        k = 0
        p = 1
        q = 2
        for i in range(n - 2):
            k = p + q
            p = q
            q = k
        return k

    def deleteDuplicates(self, head):
        """
        83删除排序链表中的重复元素
        :type head: ListNode
        :rtype: ListNode
        """
        r = head
        while head is not None and head.next is not None:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        return r

    def merge(self, nums1, m, nums2, n):
        """
        88合并两个有序数组
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i, j = m - 1, n - 1
        index = m + n - 1
        while i > -1 and j > -1:
            if nums1[i] < nums2[j]:
                nums1[index] = nums2[j]
                j -= 1
                index -= 1
            else:
                nums1[index] = nums1[i]
                i -= 1
                index -= 1
        while j > -1:
            nums1[index] = nums2[j]
            j -= 1
            index -= 1
        while i > -1:
            nums1[index] = nums1[i]
            i -= 1
            index -= 1

    def isSameTree(self, p, q):
        """
        100相同的树
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if q is None and p is None:
            return True
        if q is None or p is None:
            return False
        if p.val != q.val:
            return False
        else:
            return  self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSymmetric(self, root):
        """
        101对称二叉树
        :type root: TreeNode
        :rtype: bool
        """
        return self.ismirror(root, root)

    def ismirror(self, left, right):
        """
        :param left: TreeNode
        :param right: TreeNode
        :return: bool
        """
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        if left.val != right.val:
            return False
        else:
            return self.ismirror(left.left, right.right) and self.ismirror(left.right, right.left)

    def maxDepth(self, root):
        """
        104二叉树的最大深度
        :type root: TreeNode
        :rtype: int
        """
        return self.deep(root, 0)

    def deep(self, root, max_depth):
        """
        :param root: TreeNode
        :param max_depth: int
        :return: int
        """
        if root is None:
            return max_depth
        else:
            max_depth += 1
            max_depth = max(self.deep(root.left, max_depth), self.deep(root.right, max_depth))
            return max_depth

    def minDepth(self, root):
        """
        111二叉树的最小深度
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            return self.deep1(root, 0)

    def deep1(self, root, min_depth):
        """
        :param root: TreeNode
        :param max_depth: int
        :return: int
        """
        if root.left is None and root.right is None:
            return min_depth + 1
        elif root.left is None and root.right is not None:
            min_depth += 1
            min_depth = self.deep(root.right, min_depth)
        elif root.right is None and root.left is not None:
            min_depth += 1
            min_depth = self.deep(root.left, min_depth)
        else:
            min_depth += 1
            min_depth = min(self.deep(root.left, min_depth), self.deep(root.right, min_depth))
        return min_depth

    def singleNumber(self, nums):
        """
        136只出现一次的数字
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for x in nums:
            a = a ^ x
        return a

    def nextGreaterElements(self, nums):
        """
        503下一个更大元素2
        应使用单调栈和循环数组，未学会
        :type nums: List[int]
        :rtype: List[int]
        """
        lenth = len(nums)
        nums = nums + nums
        flag = 1
        result = []
        for i in range(lenth):
            for j in range(lenth):
                if nums[i] < nums[j + i]:
                    result.append(nums[i + j])
                    flag = 0
                    break
            if flag == 1:
                result.append(-1)
            else:
                flag = 1
        return result

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    s = Solution()
    print(s.climbStairs(4))

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
