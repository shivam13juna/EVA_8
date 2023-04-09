import os

def modify_file(filename, new_number):
    with open(filename, 'r') as file:
        lines = file.readlines()
    try:
        for i in range(len(lines)):
            line_parts = lines[i].split(' ', 1)
            line_parts[0] = str(new_number)
            lines[i] = ' '.join(line_parts)

        with open(filename, 'w') as file:
            file.writelines(lines)
    except:
        pass

dir_path = '.'  # Replace '.' with the path to your directory, if needed

for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        if filename.startswith('Anna'):
            modify_file(os.path.join(dir_path, filename), 1)
        elif filename.startswith('Elsa'):
            modify_file(os.path.join(dir_path, filename), 0)
        elif filename.startswith('Olaf'):
            modify_file(os.path.join(dir_path, filename), 2)
        elif filename.startswith('Kristoff'):
            modify_file(os.path.join(dir_path, filename), 3)
