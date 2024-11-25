# Testing space

from vmatplot.commons import process_boundary

ia = 4
ib = (-4,6)

ra = process_boundary(ia)
rb = process_boundary(ib)

print(ra)
print(rb)
