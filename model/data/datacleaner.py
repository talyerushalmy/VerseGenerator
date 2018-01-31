import re

# read raw_data.txt and ignore non-ascii characters
f = open('raw_data.txt', 'r+')
raw_data = f.read().decode('utf-8').encode('ascii', 'ignore')
f.close()

# use unix line feed instead of windows or mac
new_data = raw_data.replace('\r\n', '\n')
new_data = raw_data.replace('\r', '\n')

# clear consecutive line feeds and spaces
new_data = re.sub(r'\n\s*', '\n', new_data)
new_data = ' '.join(new_data.split(' '))

# write result to data.txt
f_out = open('data.txt', 'w+')
f_out.write(new_data)
f_out.close()