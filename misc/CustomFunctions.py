# if stepper is true, make a list of clips
def LoadData(filename, isStepper=False):
    preData = []
    data = []
    
    with open("/home/pau1o-hs/Documents/Database/" + filename + ".txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]

            if inner_list == ['']:
                if isStepper:
                    data.append(preData)
                    preData = []
                continue

            converted = []
            for item in inner_list: converted.append(float(item))

            preData.append(converted)
    
    if not isStepper:
        data = preData

    return data

def NormalizeData(data):
    means = data.mean(dim=1, keepdim=True)
    stds  = data.std(dim=1, keepdim=True)
    normalized_data = (data - means) / stds

    return normalized_data