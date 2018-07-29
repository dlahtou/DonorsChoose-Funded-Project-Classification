with open('data/Projects.csv', 'r') as open_file:
    headers = open_file.readline().split(',')
    x = open_file.readline().split(',')
    print(x)
    print(x[-1])

headers = [i.lower().replace(' ', '_') for i in headers]
print(', '.join(headers))