import os
package_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
print(package_name)