# How 2 Use
![image](https://user-images.githubusercontent.com/47798805/190232100-e8a566fc-2725-4257-a1e1-286b591351b3.png)

## 1. Create CNN
![image](https://user-images.githubusercontent.com/47798805/190232129-35245ce5-5238-421c-bde3-535dde7fd8f8.png)

## 2. Create DNN
![image](https://user-images.githubusercontent.com/47798805/190232174-a5fda96d-ff48-4fd8-a4c2-e01593a9521a.png)

## 3. Create Model
![image](https://user-images.githubusercontent.com/47798805/190232269-bbd67d99-8dc7-411c-b805-667d28674fe1.png)

## 4. Load shared library function (v4.0.0 is only work for window now)
![image](https://user-images.githubusercontent.com/47798805/190232299-17d8ba99-674a-4031-a747-efac628267ea.png)

## 5. Cropping the display of area
![image](https://user-images.githubusercontent.com/47798805/190232373-b55eef4b-b9df-46f6-9e34-531f727fbeef.png)

## 6. Create Tree
![image](https://user-images.githubusercontent.com/47798805/190232413-d561f63d-c8ca-4916-ac33-563461c5ad5d.png)

## Add : add a new node
![image](https://user-images.githubusercontent.com/47798805/190232934-965f71bb-0226-4f27-8398-79992f0dde0f.png)

## Mouse right click : Edit the name of the node (if the file name is exist the color of the node is automatically changed)
![image](https://user-images.githubusercontent.com/47798805/190232978-ac57445c-8449-4433-b00b-b23c8493e939.png)
![image](https://user-images.githubusercontent.com/47798805/190233034-055c513d-43a4-4a25-b4f6-a85bc9ac7e7e.png)
### If the name of the node is p or x then it is automatically changed as the operation block + and x
![image](https://user-images.githubusercontent.com/47798805/190233707-280f3656-7bdb-4128-8f4a-22851b03216d.png)
- operation block means the output of operation block is the smae as all input values by operations + or x

## Mouse left click : move the location of clicked node or connect the node from clicked node to the last clicked node
![image](https://user-images.githubusercontent.com/47798805/190233183-9b5fc21d-ee81-4852-a9dd-93e8f2aedf9a.png)
![image](https://user-images.githubusercontent.com/47798805/190233330-22a94dc7-bbdd-4560-9b2a-6d571d197b2e.png)

## Mouse left double click : disconnect all of its connection
![image](https://user-images.githubusercontent.com/47798805/190234104-870381ec-f87e-4c7f-9524-02e690044069.png)
![image](https://user-images.githubusercontent.com/47798805/190234132-7b0b547c-5d7b-4703-98ce-47d17bf420a0.png)

## Delete Key : delete the block
![image](https://user-images.githubusercontent.com/47798805/190234189-57dbe3a7-6878-47cb-a3cd-4ba70635a3fc.png)

## Save : After the click the node then save its all children
## Load : load the tree file

## 7. open terminal then ``` usg_AI.exe ---run t [tree name] [max iteration] [save file name]```
## EX. open terminal then move to the path that you installed 
```cd game```, ``` pytho game.py```, ```cd ..```, then move the pygame window on the right top of displayer ``` usg_AI.exe --run t ./save/tree.txt 500 ./save/result.txt``` 
