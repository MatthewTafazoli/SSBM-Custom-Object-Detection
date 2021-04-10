def convert_classes(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + " id: " + str(id) + "\n"
        msg = msg + " name: '" + name + "'\n}\n\n"
    return msg[:-1]
#Change the IDs as necessary
label_map = convert_classes(['Fox', 'Falco', 'Falcon', 'Marth'])
output_path = "./bboxredo/utils/"
with open(output_path + "label_map.pbtxt", "w") as f:
    f.write(label_map)
    f.close()
