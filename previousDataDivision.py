# making the directories

# os.makedirs(Pos_Path)
# os.makedirs(Neg_Path)
# os.makedirs(Anc_Path)

# #moving all the labelled faces in the wild images into the "negative" folder
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)): #from each folder in the lfw move all the photos
#         Ex_Path = os.path.join('lfw', directory, file)
#         New_Path = os.path.join(Neg_Path, file)
#         os.replace(Ex_Path, New_Path)
