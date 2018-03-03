import re

# read raw_data.txt and ignore non-ascii characters
f = open('raw_data.txt', 'r+')
raw_data = f.read().decode('utf-8').encode('ascii', 'ignore')
f.close()

# use unix line feed instead of windows or mac
new_data = raw_data.replace('\r\n', '\n')
new_data = new_data.replace('\r', '\n')

# clear consecutive line feeds and spaces
new_data = re.sub(r'\n\s*', '\n', new_data)
new_data = ' '.join(new_data.split(' '))

# data that will be written to the output file (data.txt)
long_line = ''

# remove leading and trailing apostrophes
for line in new_data.split('\n'):
	if line.startswith('"'):
		line = line[1::]
	if line.endswith('"'):
		line = line[::1]
	long_line += '%s ' % line

# clear consecutive spaces
long_line = re.sub(' +', ' ', long_line)

# write result to data.txt
f_out = open('data.txt', 'w+')
f_out.write(long_line)
f_out.close()