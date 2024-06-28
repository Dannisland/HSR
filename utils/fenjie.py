import math

def count_substring_with_wildcards(mainstr, substr):
    head, mid, tail = substr.split('?')[0], substr.split('?')[1], substr.split('?')[2]

    head_fre = mainstr.count(head)
    mid_fre = mainstr.count(mid)
    tail_fre = mainstr.count(tail)

    if head_fre == 0 or mid_fre == 0 or tail_fre == 0:
        return 0

    head_list = find_occurrences(mainstr, head)
    mid_list = find_occurrences(mainstr, mid)
    tail_list = find_occurrences(mainstr, tail)

    count = 0

    for i in head_list:
        for j in mid_list:
            for k in tail_list:
                if i < j < k:
                    count += 1

    return count

def find_occurrences(mainstr, pattern):
    occurrences = []
    start = 0
    while True:
        index = mainstr.find(pattern, start)
        if index == -1:
            break
        occurrences.append(index)
        start = index + 1
    return occurrences

# 示例输入
mainstr = str(input()) # "12323454556767"
substr = str(input()) # "23?45?67"

# 调用函数并打印结果
result = count_substring_with_wildcards(mainstr, substr)
print(result)  # 输出: 8
