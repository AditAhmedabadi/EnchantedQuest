import os

def chat_template(lst):
    convo_template = []

    for i, message in enumerate(lst):
        message_dict = {'role': None, 'content': None}
        if "[Player]" in message:
            message_dict['role'] = 'user'
            # message = message.replace("[Player]:","")
            message_dict['content'] = str(message)
            convo_template.append(message_dict)        
        else:
            message_dict['role'] = 'assistant'
            message_dict['content'] = str(message)
            convo_template.append(message_dict)
    return convo_template

def add_game_start(convo):
    init_text = 'Play a game with me where i am in an enchanted forest full of beasts and loots. You be the narrator and i will be the player, play a dialog game with me.'
    init_role = {'role':'user','content':init_text}
    convo.insert(0,init_role)
    return convo

def list_convo(text_file, quiet):
    with open(f"data/{text_file}",'r') as f:
        data = f.read()
    lst = data.split('\n\n')
    combined_items = str(lst[-2] + lst[-1])
    lst[-2:] = [combined_items]
    # print(len(lst))

    convo_template = chat_template(lst)
    if quiet == False:
        if len(lst) == len(convo_template):
            print(len(lst),"| Successful Template Conversion")
    return convo_template

def template_from_dir(directory, quiet = True):
    convo_dataset = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            convo_dataset.append(add_game_start(list_convo(filename,quiet = quiet))) 
    return convo_dataset
    
if __name__ == "__main__":
   convo_dataset = template_from_dir("data", quiet = True)
   print(convo_dataset[2],'\n\n' ,len(convo_dataset))