from torchvision import transforms

# torch transforms
def ts(object):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transform(object)