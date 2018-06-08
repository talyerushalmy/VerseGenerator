import re

# read raw_data.txt and ignore non-ascii characters
f = open('raw_data.txt', 'r')
raw_data = f.read().decode('utf-8').encode('ascii', 'ignore')
f.close()

# use unix line feed instead of windows or mac
new_data = raw_data.replace('\r\n', '\n')
new_data = new_data.replace('\r', '\n')

# clear consecutive line feeds and spaces
new_data = re.sub(r'\n\s*', '\n', new_data)
new_data = ' '.join(new_data.split(' '))

# clear verse numbers
new_data = re.sub(r'\\[0-9]+:[0-9]+\\', '', new_data)

# data that will be written to the output file (data.txt)
long_line = ''

# remove leading and trailing apostrophes
for line in new_data.split('\n'):
	line = ''.join([c for c in line if c.isalnum() or c in ' ,\n."\':;'])
	line = line.strip()
	if line.startswith('"'):
		line = line[1:]
	if line.endswith('"'):
		line = line[:-1]
	long_line += '%s ' % line

# clear consecutive spaces
long_line = re.sub(' +', ' ', long_line)

# break lines at dots
output_data = long_line.replace('."', '\n\n')
output_data = output_data.replace('.', '.\n')
output_data = output_data.replace('\n\n', '."\n')

# remove leading spaces
output_data = '\n'.join([line.strip() for line in output_data.split('\n')])

# change all characters to lower case
output_data = output_data.lower()

# write result to data.txt
f_out = open('data.txt', 'w+')
f_out.write(output_data)
f_out.close()