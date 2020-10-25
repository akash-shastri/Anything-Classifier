

# In[5]:


from fastai.vision.all import *
from fastai.vision.widgets import *
from tkinter import *
import shutil
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk


# ## Imports

# In[6]:


import sys
from Scraping.scraper import *



# ## Scraping Images
# 

# In[7]:



def Create_db(DRIVER_PATH, search_term, number_images = 100, target_path = './images'):
    for ele in search_term:
        search_and_download(search_term = ele, driver_path=DRIVER_PATH,  number_images=number_images, target_path=target_path)


# ## Training
# 

# In[28]:


def train(path = None):
    
    data = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128),)



    data = data.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
    dls = data.dataloaders(path, num_workers = 0)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)

    learn = cnn_learner(dls, resnet34, metrics=error_rate, num_workers = 0)
    learn.fine_tune(5)


    learn.export()

    path = Path()
    path.ls(file_exts='.pkl')


# ## Inference

# In[38]:



def Classify():
    
    learn_inf = load_learner('./export.pkl')
    root = Toplevel()
    root.geometry("500x500")
    def UploadAction(event=None):
        global img
        global label
        
        filename = filedialog.askopenfilename()
        img = PILImage.create(filename)
        load = PIL.Image.open(filename)
        load = load.resize((300,300))
        render = PIL.ImageTk.PhotoImage(load)
        label = Label(root, image=render)
        label.image = render
        label.grid(row = 0, column = 0, columnspan = 3)
        
    def ClassifyAction(event=None):
        global label2
        pred,pred_idx,probs = learn_inf.predict(img)
        label2 = Text(root, height=2, width=30)
        label2.grid(row = 1, column = 0, columnspan = 3)
        label2.insert(INSERT, f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')        
        
    def ClearAction():
        label.grid_forget()
        label2.grid_forget()
    up_button = Button(root, text='Upload', command = UploadAction)
    up_button.grid(row = 2, column = 0)
    
    clas_button = Button(root, text = 'Classify', command = ClassifyAction)
    clas_button.grid(row =2, column = 1)
    
    clear_button = Button(root, text = 'Clear', command = ClearAction)
    clear_button.grid(row =2, column = 2)
    
    root.mainloop()


# # RUN
# 

# In[59]:


top = Tk()
top.title("Main")
top.geometry("200x200")
global target_path_i
target_path_i =  './images'
def on_create_ds():
    global target_path_i
    root = Toplevel(height = 500, width = 500)
    root.title("Create Database")
    def retrieve_input():
        global target_path_i
        Driver_path_i=Driver_path.get("1.0","end-1c")
        options=search_term.get(1.0, END)# get lines into string
        #Remove spaces in list of lines
        search_term_i = [i.strip() for i in options.splitlines()]
        
        number_images_i=number_images.get("1.0","end-1c")
        target_path_i=target_path.get("1.0","end-1c")
        try:
            shutil.rmtree(target_path_i)
        except:
            pass
        Create_db(Driver_path_i, search_term_i, number_images = 100, target_path= target_path_i)
        
        label_done = Label(root, text = "DONE! YOU CAN TRAIN NOW! ")
        label_done.pack(padx = 10, pady = 10)

    var1 = StringVar()
    label1 = Label( root, textvariable=var1 )
    var1.set("Driver Path")
    label1.pack()
    Driver_path=Text(root, height=2, width=100)
    Driver_path.pack(padx = 10, pady = 10)
    Driver_path.insert(INSERT,"D:\Deep Learning\Scraping\chromedriver.exe")
    
    var2 = StringVar()
    label2 = Label( root, textvariable=var2 )
    var2.set("search_term")
    label2.pack()
    search_term=Text(root, height=2, width=100)
    search_term.pack(padx = 10, pady = 10)
    
    var3 = StringVar()
    label3 = Label( root, textvariable=var3 )
    var3.set("number_images")
    label3.pack()
    number_images=Text(root, height=2, width=100)
    number_images.pack(padx = 10, pady = 10)
    number_images.insert(INSERT,"100")
    
    var4 = StringVar()
    label4 = Label( root, textvariable=var4 )
    var4.set("target_path")
    label4.pack()
    target_path=Text(root, height=2, width=100)
    target_path.pack(padx = 10, pady = 10)
    target_path.insert(INSERT,"./images")
    
    buttonCommit=Button(root, height=1, width=10, text="Commit", 
                    command=lambda: retrieve_input())
    buttonCommit.pack()
    root.mainloop()
 
# Add Changes here to allow hyperparameter tuning   
def on_train():
    
    train(path = str(target_path_i))
    label = Label(top, text = "DONE! You can Classify NOW!")
    label.grid(row = 3, column = 0)
    
    



b1 = Button( text ="create_ds", command = on_create_ds)
b1.grid(row = 0, column = 0)
b2 = Button(text ="train", command = on_train)
b2.grid(row = 1, column = 0)
b3 = Button( text ="classify", command = Classify)
b3.grid(row = 2, column = 0)




top.mainloop()


# In[ ]:





# In[ ]:




