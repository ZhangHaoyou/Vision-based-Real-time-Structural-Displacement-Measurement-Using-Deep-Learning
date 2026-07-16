with open('./logs/positions.txt', 'r') as txt:
    numbers = []
    lines = txt.readlines()
    for i, line in enumerate(lines):
        if line == 'None\n':
            numbers.append(i-1)

with open('./numbers.txt', 'w') as num:
    num.write('[')
    for number in numbers:
        num.write(f'{number}, ')
    num.write(']')