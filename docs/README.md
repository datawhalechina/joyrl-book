# JoyRL Book

## 几个问题

**为什么要做`JoyRL Book`？**

`JoyRL Book` 侧重于帮助读者快速入门强化学习的代码实践，并辅以一套开源代码框架，便于读者适应业界应用研究风格的代码。

**与[蘑菇书](https://github.com/datawhalechina/easy-rl)的区别?**

* **理论深度不同**：蘑菇书侧重更详细更通俗的理论讲解，适合细嚼慢咽的读者，`JoyRL Book`则致力于方便应用的核心理论，讲解相对更加简明，适合具有一定数学基础且希望快速进入实践应用的读者。

* **代码实战不同**：蘑菇书以`Jupyter Notebook`形式讲解基础的算法，`JoyRL Book` 则配套一个更完整的代码生态，具体见关于`JoyRL`部分。`JoyRL Book`与蘑菇书各有侧重点，读者可根据自身情况按需择取。

**关于`JoyRL`?**

`JoyRL`旨在建立一套帮助初学者或交叉学科研究者快速入门强化学习的代码生态，主要包括`JoyRL离线版`，`JoyRL在线版`，`JoyRL论文`等几大部分：

* [JoyRL离线版](https://github.com/johnjim0816/joyrl-offline)：离线版开源框架。保留每个算法的完整结构，便于读者学习使用，配以中文注释，适合读者学习使用。在此基础上，编写完整的框架（例如配置多线程）帮助读者进行强化学习的高效应用。同时也是开发版框架，在开发新的算法时首先会在离线版中测试，然后同步到`JoyRL在线版`。
* [JoyRL在线版](https://github.com/datawhalechina/joyrl)：以`PiP`包的形式开发开源框架，英文注释，会比离线版更加集成，更加高效，并且会去掉一些实际并不常用的基础算法，例如`Q-learning`等等，适合需要大规模环境应用的读者进阶使用
* [JoyRL论文](https://github.com/datawhalechina/rl-papers)：定时收集强化学习各类子方向的前沿论文，帮助读者快速了解相关领域的研究
  
## 内容目录

代码实战请转到[JoyRL离线版](https://github.com/johnjim0816/joyrl-offline)或者[JoyRL在线版](https://github.com/datawhalechina/joyrl)



|               章节                | 关键内容 |
| :-------------------------------: | :--: |
|       [第一章 绪论](/ch1/main.md)       | 待更新 |
| [第二章 马尔可夫决策过程](/ch2/main.md) | 马尔可夫决策过程、状态转移矩阵 |
|     [第三章 动态规划](/ch3/main.md)     | 贝尔曼方程、策略迭代、价值迭代 |
|    [第四章 免模型预测](/ch4/main.md)    | 蒙特卡洛、时序差分 |
|    [第五章 免模型控制](/ch5/main.md)    | Q-learning 算法、Sarsa 算法 |
|    [第六章 深度学习基础](/ch6/main.md)    | 待更新 |
|    [第七章 DQN算法](/ch7/main.md)    | 目标网络、经验回放 |
|    [第八章 DQN算法进阶](/ch8/main.md)    | Double DQN、Dueling DQN、 |
|    [第九章 策略梯度](/ch9/main.md)    |      |
|    [第十章 Actor-Critic 算法](/ch10/main.md)    |      |
|    [第十一章 DDPG 算法](/ch11/main.md)    |      |
|    [第十二章 PPO 算法](/ch12/main.md)    |      |
|    [第十三章 SAC 算法](/ch13/main.md)    |      |

## 贡献者


<table border="0">
  <tbody>
    <tr align="center" >
        <td>
         <a href="https://github.com/JohnJim0816"><img width="70" height="70" src="https://github.com/JohnJim0816.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/JohnJim0816">John Jim</a>
         <p>教程设计与算法实战<br> 北京大学硕士</p>
        </td>
        <td>
            <a href="https://github.com/qiwang067"><img width="70" height="70" src="https://github.com/qiwang067.png?s=40" alt="pic"></a><br>
            <a href="https://github.com/qiwang067">Qi Wang</a> 
            <p>教程设计<br> 上海交通大学博士生<br> 中国科学院大学硕士</p>
        </td>
        <td>
            <a href="https://github.com/yyysjz1997"><img width="70" height="70" src="https://github.com/yyysjz1997.png?s=40" alt="pic"></a><br>
            <a href="https://github.com/yyysjz1997">Yiyuan Yang</a> 
            <p>教程设计 <br> 牛津大学博士生<br> 清华大学硕士</p>
        </td>
    </tr>
  </tbody>
</table>