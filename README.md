# wechat_jump
This program can set a goal score and recognize the score of every step during playing. When reaching the goal, it will ask the user whether to exit the game. If not, user can reset a new goal and continue playing.
## Part one:
1, Generate the numbers page manually (eg, the **numbers_1440_2560.jpg**).</br>
2, Run **knn_model.py**. Then user need to enter the corresponding number after the number is marked.
## Part two:
1, Cut out the chess pic(eg, the **chess_1440_2560.jpg**).</br>
2, Run **wechat_jump.py**. When the score reach the goal, the program will ask you whether to exit the game.
### Environment:
opencv  3.3.1 </br>
python  3.5 </br>
numpy   1.13.3 </br>
### Tips:
This version is able to work on resolution ratio 1440*2560. If the resolution is different, just replace the number pic and chess pic.
