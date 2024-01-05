from sen2venus import Sen2Venus, Sen2VenusSite
import matplotlib.pyplot as plt
import fire

def test_concat_class(idx=0):
    dataset = Sen2Venus('./', ['SUDOUE-4', 'FGMANAUS'], load_geometry=True, subset='all')
    print(dataset)

if __name__ == '__main__':
    fire.Fire(test_concat_class)
    