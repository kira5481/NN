#about torch...
# импорт основного модуля PyTorch
import torch 
# импорт модуля нейронных сетей
import torch.nn as nn 
# импорт оптимизаторов
import torch.optim as optim 
# импорт функций активации
import torch.nn.functional as F  

# импорт датасетов, моделей и трансформаций из torchvision
from torchvision import datasets, models, transforms 
# импорт загрузчика данных и класса Dataset
from torch.utils.data import DataLoader, Dataset 
# импорт матрицы ошибок
from sklearn.metrics import confusion_matrix  
# импорт класса для визуализации матрицы ошибок
from sklearn.metrics import ConfusionMatrixDisplay 

#using numpy
#using numpy

# импорт библиотеки numpy для работы с массивами
import numpy as np 

#for data load or save -  импорт библиотеки pandas для работы с таблицами
import pandas as pd 

#visualize some datasets -  импорт библиотеки для визуализации данных
import matplotlib.pyplot as plt 

#check our work directory -  импорт модуля для работы с операционной системой
import os 


# learning_rate   - задаем шаг обучения
lr = 0.001 
# we will use mini-batch method - задаем размер мини-батча
batch_size = 100 

    #{INF} - Batch size (батч сайз) - это количество образцов данных, которые обрабатываются за один проход алгоритмом обучения (например, нейронной сетью), перед обновлением весов.
    #В данном случае, batch_size = 100 означает, что модель будет обрабатывать 100 образцов данных одновременно перед тем, как производить обновление весов на основе их ошибок.
    #Выбор оптимального размера батча зависит от многих факторов, таких как доступная память на GPU, сложность модели, размер обучающей выборки и т.д. 
    #Больший размер батча может ускорить процесс обучения, но также может привести к более шумной и менее точной оценке градиента, а слишком маленький размер батча 
    #может привести к нестабильности в обучении.

# How much to train a model - количество эпох обучения модели
epochs = 12 
# задаем флаг обучения модели
train = True 
# os.listdir('data/train') - проверка содержимого директории

# определение устройства, на котором будет проходить обучение
device = 'cuda'  

# инициализация генератора случайных чисел для воспроизводимости результатов
torch.manual_seed(1234)  
# включение использования операций матричного умножения с 32-битной точностью
torch.backends.cuda.matmul.allow_tf32 = True 
# включение использования оптимизированных алгоритмов для быстрого выполнения вычислений на GPU
torch.backends.cudnn.allow_tf32 = True 

# проверяем, что CUDA доступна на устройстве
if device == 'cuda': 
    # инициализация генератора случайных чисел для воспроизводимости результатов
    torch.manual_seed(1234) 

    # torch.mps.manual_seed_all(1234)  # инициализация генератора случайных чисел для использования многопроцессорного режима
    # os.makedirs('data/catORdog', exist_ok=True) # создание директории для хранения обученной модели

    # путь к директории с обучающими данными
    train_dir = 'train/train' 
    # путь к директории с тестовыми данными
    test_dir = 'test/test' 
    # список для хранения путей к файлам обучающей выборки
    train_list = [] 
    # список для хранения путей к файлам тестовой выборки
    test_list = [] 

    #импортируем модуль Path из библиотеки pathlib
    from pathlib import Path 
    
    # добавляем в список train_list пути к изображениям *.jpg в директории train_dir и ее поддиректориях
    for path in Path(train_dir).rglob('*.jpg'):
        train_list.append(path)
    
    # добавляем в список test_list пути к изображениям *.jpg в директории test_dir и ее поддиректориях
    for path in Path(test_dir).rglob('*.jpg'): 
        test_list.append(path)

    # импортируем модуль Image из библиотеки PIL
    from PIL import Image 

        #{INF} - PIL - это библиотека Python для обработки изображений. Она предоставляет множество инструментов для работы с изображениями, 
        #включая изменение размеров, обрезание, наложение фильтров и преобразование цветовых пространств.
        #Когда мы импортируем модуль Image из библиотеки PIL, мы можем использовать класс Image для загрузки, обработки и сохранения изображений.

    #генерируем 10 случайных чисел от 1 до 25000
    random_idx = np.random.randint(1, 25000, size=10) 

    # импортируем функцию train_test_split из библиотеки sklearn.model_selection
    from sklearn.model_selection import train_test_split 
    #разделяем список train_list на две части: train_list и val_list в соотношении 80/20
    train_list, val_list = train_test_split(train_list, test_size=0.2) 
    
    # data Augumentation - преобразование данных для увеличения их разнообразия и предотвращения переобучения

    # определяем трансформации для обучающих данных, которые будут применяться при каждой эпохе обучения
    train_transforms = transforms.Compose([ 
        # изменяем размер изображения до 224х224
        transforms.Resize((224, 224)), 
        # случайно обрезаем изображение до размера 224x224
        transforms.RandomResizedCrop(224),
        # случайно отражаем изображение по горизонтали
        transforms.RandomHorizontalFlip(), 
        # преобразуем изображение в тензор
        transforms.ToTensor(), 
    ])
    # определяем трансформации для проверочных данных, которые будут применяться при каждой эпохе обучения
    val_transforms = transforms.Compose([ 
        # изменяем размер изображения до 224х224
        transforms.Resize((224, 224)), 
        #случайно обрезаем изображение до размера 224x224
        transforms.RandomResizedCrop(224), 
        #случайно отражаем изображение по горизонтали
        transforms.RandomHorizontalFlip(), 
        # преобразуем изображение в тензор - значения в диапазоне [0,1]
        transforms.ToTensor(),
    ])

    # определяем трансформации для тестовых данных, которые будут применяться при оценке модели
    test_transforms = transforms.Compose([ 
        # изменяем размер изображения до 224х224
        transforms.Resize((224, 224)), 
        #случайно обрезаем изображение до размера 224x224
        transforms.RandomResizedCrop(224), 
        #случайно отражаем изображение по горизонтали
        transforms.RandomHorizontalFlip(), 
        # преобразуем изображение в тензор - значения в диапазоне [0,1]
        transforms.ToTensor() 
    ])


    # Создание класса датасета с методами для загрузки изображений из файла
    class dataset(torch.utils.data.Dataset): 

        # Конструктор класса с параметрами: список файлов и преобразования (transformations)
        def __init__(self, file_list, transform=None): 
            # Список файлов входных изображений
            self.file_list = file_list 
           
            # Набор преобразований для каждого изображения
            self.transform = transform 

        # dataset length # Длина датасета
        def __len__(self):
            self.filelength = len(self.file_list)
            return self.filelength

        # load an one of images - Загрузка изображения
        def __getitem__(self, idx):
            # Получаем путь к изображению
            img_path = self.file_list[idx] 

            # Загружаем изображение
            img = Image.open(img_path)  
            # Применяем к изображению набор трансформаций
            img_transformed = self.transform(img) 
            # Получаем метку класса
            label = img_path.parts  

            # Определяем метку класса для каждого изображения

            #Проверяем, находится ли изображение в папке train или test
            if img_path.parts[1] == 'train': 

                #Если изображение находится в папке train, то из названия файла получаем 
                #метку класса (кошка или собака) и присваиваем числовое значение 0 для кошек и 1 для собак
                if label[2].split('.')[0] == 'cat':
                    label = 0
                elif label[2].split('.')[0] == 'dog':
                    label = 1

               
            else:
                #Если изображение находится в папке test, то из названия файла получаем метку класса и сохраняем ее:
                label = label[2].split('.')[0]

            return img_transformed, label # Возвращаем преобразованное изображение и его метку

   
    # Создаем объекты датасетов для обучения, тестирования и валидации
    train_data = dataset(train_list, transform=train_transforms) 
    test_data = dataset(test_list, transform=test_transforms)
    val_data = dataset(val_list, transform=test_transforms)
    
    # Создаем объекты загрузчиков данных для обучения, тестирования и валидации
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

        #{INF} - Код создает объекты датасетов для обучения, тестирования и валидации с помощью определенных списков изображений и преобразований, 
        #а затем использует их для создания загрузчиков данных с помощью torch.utils.data.DataLoader. Это позволяет разбивать данные на мини-пакеты, 
        #что упрощает обучение нейронной сети. Параметр batch_size определяет размер мини-пакета, а параметр shuffle позволяет перемешивать данные при каждой эпохе.

   

    class Cnn(nn.Module): # Определяем конструктор класса
        def __init__(self):
            super(Cnn, self).__init__() # Вызываем конструктор класса nn.Module для класса Cnn
            

            #первый блок содержит сверточный слой с ядром размера 3x3, 16 выходными каналами (фильтрами) и шагом (stride) 2, 
            #что означает, что каждый второй пиксель будет пропущен. Затем следует слой Batch Normalization, функция активации ReLU 
            #и слой MaxPooling с размером ядра 2x2.
            self.layer1 = nn.Sequential(   # Определяем слои свертки и пулинга
                nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            #второй блок имеет сверточный слой с 32 выходными каналами и аналогичными параметрами, 
            #что и в первом блоке, затем следует Batch Normalization, ReLU и MaxPooling.
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            #третий блок имеет сверточный слой с 64 выходными каналами и аналогичными параметрами, 
            #что и предыдущие два блока, затем Batch Normalization, ReLU и MaxPooling.
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            # Определяем полносвязные слои

            #первый полносвязный слой, который имеет входной размер 3 * 3 * 64 и выходной размер 254.
            self.fc1 = nn.Linear(3 * 3 * 64, 254)
            #слой Dropout для регуляризации.
            self.dropout = nn.Dropout(0.5)
            #второй полносвязный слой, который принимает на вход 254 и выдает на выход 120.
            self.fc2 = nn.Linear(254, 120)
            #третий полносвязный слой, который принимает на вход 120 и выдает на выход 80.
            self.fc3 = nn.Linear(120, 80)
            #четвертый полносвязный слой, который принимает на вход 80 и выдает на выход 40.
            self.fc4 = nn.Linear(80, 40)
            #пятый и последний полносвязный слой, который принимает на вход 40 и выдает на выход 2 (так как у нас только два класса - кошки и собаки).
            self.fc5 = nn.Linear(40, 2)
            
            
            
        # Определяем метод для прямого распространения сигнала в нейронной сети
        def forward(self, x):
            # Пропускаем входные данные через слои свертки и пулинга
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            # Выпрямляем выходные данные слоев свертки и пулинга
            out = out.view(out.size(0), -1)
            # Пропускаем данные через полносвязные слои
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            out = self.fc5(out)
            # Возвращаем выходные данные
            return out

    #Создаем экземпляр модели Cnn и отправляем его на устройство device (GPU, если он доступен).
    model = Cnn().to(device) 

    #Проверяем, существует ли файл с названием "CNNmodel.pt".
    if os.path.isfile("CNNmodel.pt"): 
        #Если файл существует, загружаем веса модели из этого файла в model.
        model.load_state_dict(torch.load('CNNmodel.pt')) 
    #Переводим модель в режим обучения.
    model.train() 
    #Определяем оптимизатор Adam и передаем ему параметры модели model для оптимизации с заданным коэффициентом обучения lr.
    optimizer = optim.Adam(params=model.parameters(), lr=lr) 
    #Определяем функцию потерь, которую будем использовать во время обучения модели.
    criterion = nn.CrossEntropyLoss()

    #инициализируем переменные lastAccuracy и nowAccuracy

    #lastAccuracy - используется для хранения значения точности (accuracy) модели на предыдущей эпохе.
    lastAccuracy = 0 
    #nowAccuracy - используется для хранения значения точности (accuracy) модели на текущей эпохе. 
    nowAccuracy = 0

    #проверяем, если train=True, то начинаем обучение
    if train == True :
        print("Train Starts") # выводим сообщение, что обучение началось
        # создаем пустые массивы для хранения истинных и предсказанных меток и для значений функции потерь
        y_true = []
        y_false = []
        loss_arr = []

        # запускаем цикл обучения на определенное количество эпох
        for epoch in range(epochs):
            # выводим сообщение, что началась новая эпоха
            print("Epoch Starts")
            # инициализируем переменные для хранения значения функции потерь и точности на данной эпохе
            epoch_loss = 0
            epoch_accuracy = 0


            # if os.path.isfile("CNNmodel.pt"):
            #     model.load_state_dict(torch.load('CNNmodel.pt'))


            # запускаем цикл обучения на каждый батч в тренировочных данных
            for data, label in train_loader:
                # отправляем батч на GPU
                data = data.to(device)  
                #Создается список temp, в котором первый элемент - это y_true, а второй элемент - это результат применения метода numpy() к label.  
                temp = [ y_true, label.numpy()]
                #Результаты из списка temp конкатенируются в один numpy массив методом np.concatenate().
                temp = np.concatenate(temp)
                #Массив temp присваивается переменной y_true.
                y_true = temp

                # отправляем метки на GPU
                label = label.to(device) 

                # прогоняем данные через модель
                output = model(data) 

                # вычисляем значение функции потерь
                loss = criterion(output, label)  

                # Эта строка извлекает предсказания модели для текущей партии данных и преобразует результат в массив numpy.
                out = output.argmax(dim=1).cpu().numpy() 

                    #output - это тензор, возвращенный моделью, содержащий предсказанные вероятности для каждого класса. .argmax(dim=1) 
                    # используется для получения индекса класса с наибольшей вероятностью для каждого элемента в этом тензоре. .cpu() используется 
                    # для переноса вычислений на процессор центрального процессора, если на данном этапе вычисления проводились на графическом процессоре, 
                    # а .numpy() преобразует результат в массив numpy.
                

                # сохраняем значения истинных меток и предсказанных меток
                temp = [ y_false, out]
                temp = np.concatenate(temp)
                y_false = temp
                    #Создаем массив temp, который содержит значения из двух других массивов: y_false и out. 
                    #Затем, используя функцию np.concatenate, значения массивов y_false и out объединяются вместе в один массив temp.
                    #Наконец, массив temp присваивается переменной y_false. Таким образом, y_false становится объединенным массивом значений из первоначальных массивов.

                # обнуляем градиенты
                optimizer.zero_grad()

                # обратное распространение ошибки
                loss.backward()

                # обновляем веса
                optimizer.step()

                # вычисляем точность и значение функции потерь на данном батче

                #определяет точность текущей итерации путем сравнения выходов модели (output) с метками классов (label), используя метод argmax(dim=1)
                #для определения предсказанного класса. После этого вычисляется среднее значение (mean) точности путем преобразования булевого значения 
                # в число типа float и среднего значения по всем примерам в пакете.
                acc = ((output.argmax(dim=1) == label).float().mean())

                #учитывает текущую точность путем добавления средней точности пакета к точности за эпоху (epoch_accuracy), деленной на общее количество пакетов (len(train_loader)).
                epoch_accuracy += acc / len(train_loader)
                #вычисляет потери за эпоху, добавляя потери текущей итерации (loss), деленные на общее количество пакетов (len(train_loader)), к потерям за эпоху (epoch_loss).
                epoch_loss += loss / len(train_loader)

            # выводим значения точности и функции потерь на данной эпохе
            print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))


            #Создаёт контекстный менеджер для вычисления без градиентов, что означает, 
            # что градиенты не будут вычисляться в процессе обратного распространения ошибки в нейронной сети.
            with torch.no_grad(): 
                #Инициализация переменной для подсчета точности модели на валидационной выборке.
                epoch_val_accuracy = 0 
                #Инициализация переменной для подсчета функции потерь модели на валидационной выборке.
                epoch_val_loss = 0 
                #Цикл для обработки данных из валидационного датасета по батчам.
                for data, label in val_loader: 
                    # Перемещает данные на устройство (GPU), которое используется для вычислений.
                    data = data.to(device) 
                    # Перемещает метки на устройство (GPU), которое используется для вычислений.
                    label = label.to(device) 
                    #Прямой проход данных через модель нейронной сети.
                    val_output = model(data) 
                    # Вычисление функции потерь на валидационных данных.
                    val_loss = criterion(val_output, label) 

                    # Вычисление точности модели на текущем батче валидационных данных.
                    acc = ((val_output.argmax(dim=1) == label).float().mean()) 
                    # Обновление значения точности модели на валидационной выборке.
                    epoch_val_accuracy += acc / len(val_loader) 
                    # Обновление значения функции потерь на валидационной выборке.
                    epoch_val_loss += val_loss / len(val_loader) 

                #Обновление значения текущей точности модели на валидационной выборке.
                nowAccuracy = 100 * epoch_val_accuracy   
                # Добавление значения функции потерь на валидационной выборке в список потерь для всех эпох обучения.
                loss_arr.append(epoch_val_loss.cpu().item()) 
                # Вывод текущего значения точности и функции потерь на валидационной выборке.
                print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy, epoch_val_loss)) 
                #Вывод текущей точности модели на валидационной выборке в процентах.
                print(f'Accuracy of the network on the 10000 test images: {100 * epoch_val_accuracy} %') 

            # Если текущая точность модели на валидационной выборке больше, чем предыдущая, то:
            if(nowAccuracy > lastAccuracy): 
                # Сохранение текущего состояния модели в файл CNNmodel.pt.
                torch.save(model.state_dict(), 'CNNmodel.pt') 
                # Вывод сообщения о сохранении прогресса.
                print("Progress Saved")  
                # Обновление значения предыдущей точности модели на валидационной выборке.
                lastAccuracy = nowAccuracy 
            # Вывод списка значений функции
            print(loss_arr) 

    #Визуализация confusion matrix
    cm = confusion_matrix(y_true,y_false) 
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    #Построение графика loss-функции в зависимости от количества эпох
    plt.plot(list(range(1,epochs+1 )), loss_arr)
    
    # set the x and y labels' - Установка подписей для осей x и y
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # set the title - Установка заголовка
    plt.title('Loss vs Epochs')

    # show the plot - Отображение графика
    plt.show()

    #Предсказания модели на тестовых данных и вывод результатов в таблицу
    if train == False:
        # Создание словаря, где 0 соответствует "cat" (кошка), а 1 соответствует "dog" (собака).
        class_ = {0: 'cat', 1: 'dog'}
        #Создание пустого списка results, в который будут добавляться результаты предсказания модели.
        results = [] 
        #Установка модели в режим оценки (evaluation mode), в котором все слои работают в режиме вывода без вычисления градиентов.
        model.eval() 
        # Создание контекстного менеджера torch.no_grad(), в котором вычисление градиентов отключено. Это ускоряет процесс вывода и экономит память.
        with torch.no_grad():
            #Итерация по тестовому набору данных с помощью загрузчика данных test_loader
            for data, fileid in test_loader: 
                #Перемещение данных на устройство (GPU или CPU), которое используется для вычислений.
                data = data.to(device) 
                #Получение выходных данных модели для входных данных data.
                preds = model(data) 
                
                #Получение меток классов с наибольшими вероятностями для каждого элемента в preds.
                _, preds = torch.max(preds, dim=1) 
                
                # Добавление кортежей, состоящих из идентификатора файла (fileid) и предсказанной метки класса (preds.tolist()), в список results.
                results += list(zip(list(fileid), preds.tolist())) 
                

       
       
        # Преобразование полученных данных в формат таблицы и сохранение ее в файл

        # Создается список idx с помощью функции map, которая применяет функцию lambda x: x[0] к каждому элементу в results. 
        #Таким образом, idx будет содержать все первые элементы в кортежах из results.
        idx = list(map(lambda x: x[0], results)) 

        #Создается список prob с помощью функции map, которая применяет функцию lambda x: x[1] к каждому элементу в results. 
        #Таким образом, prob будет содержать все вторые элементы в кортежах из results.
        prob = list(map(lambda x: x[1], results))

        #Создается объект DataFrame из словаря, содержащего ключ 'id' со значением idx и ключ 'label' со значением prob. Каждому значению в списке idx соответствует значение в списке prob. 
        #Таким образом, DataFrame submission будет содержать столбец 'id' и столбец 'label'.
        submission = pd.DataFrame({'id': idx, 'label': prob})
        
        
        # Визуализация нескольких случайных изображений из тестового набора данных

        # импортирует модуль random, который содержит функции для работы с генерацией случайных чисел.
        import random 

        # создает пустой список id_list.
        id_list = [] 
        
        # создает 2 строки и 5 столбцов графиков в фигуре с белым фоном размером 5x5.
        fig, axes = plt.subplots(2, 5, figsize=(5, 5), facecolor='w') 
       
        # итерируется по осям графиков.
        for ax in axes.ravel(): 

            # выбирает случайный идентификатор из submission, сохраняет его в i.
            i = random.choice(submission['id'].values) 

            # находит метку (cat или dog) в submission, соответствующую выбранному идентификатору, сохраняет ее в label.
            label = submission.loc[submission['id'] == i, 'label'].values[0] 
            
            
            # создает путь к файлу изображения, используя идентификатор.
            img_path = os.path.join(test_dir, '{}.jpg'.format(i)) 
            #открывает файл изображения.
            img = Image.open(img_path) 
            #изменяет размер изображения до 224x224 пикселей.
            img = img.resize((224, 224)) 
            #устанавливает заголовок графика в соответствии с меткой (cat или dog).
            ax.set_title(class_[label])
            #скрывает оси графика.
            ax.axis('off') 
            #отображает изображение в графике.
            ax.imshow(img) 
           
        #показывает все графики.
        plt.show() 
    
       
        
