from sen2venus import Sen2VenusSite
import matplotlib.pyplot as plt
import fire

def test_site_class(idx=0):
    dataset = Sen2VenusSite('./', 'SUDOUE-4', load_geometry=True, subset='all')
    print(dataset)
    
if __name__ == '__main__':
    fire.Fire(test_site_class)
    