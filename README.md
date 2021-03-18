# Deep_RL_Agent_Cab_Driver
RL-based system for assisting cab drivers can potentially retain and attract new cab drivers. 

## Objective
The objective of the problem is to maximise the profit earned over the long-term.
Decision Epochs
The decisions are made at an hourly interval; thus, the decision epochs are discrete.
## Assumptions
1. The taxis are electric cars. It can run for 30 days non-stop, i.e., 24*30 hrs. Then it needs to recharge itself. If the cab-driver is completing his trip at that time, he’ll finish that trip and then stop for recharging. So, the terminal state is independent of the number of rides covered in a month, it is achieved as soon as the cab-driver crosses 24*30 hours.
2. There are only 5 locations in the city where the cab can operate.
3. All decisions are made at hourly intervals. We won’t consider minutes and seconds for this project. So for example, the cab driver gets requests at 1:00 pm, then at 2:00 pm, then at 3:00 pm and so on. So, he can decide to pick among the requests only at these times. A request cannot come at (say) 2.30 pm.
4. The time taken to travel from one place to another is considered in integer hours (only) and is dependent on the traffic. Also, the traffic is dependent on the hour-of-the-day and the day-of-the-week.
