from difflib import SequenceMatcher

def get_agent_type(agent_list):
    names = agent_list

    string2 = names[0]
    for i in range(1, len(names)):
        string1 = string2
        string2 = names[i]
        match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))

    return (string1[match.a: match.a + match.size])
