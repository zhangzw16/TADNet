pk_info = [[1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.03,'amp_limited','gauss',20,0.1,1,2e-5,0.1,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_unlimited','gauss',20,0,1,2e-5,0.02,1,50],
           [1,1000,16000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-5,0.02,1,50],
           [1,500,8000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.5,1,50],
           [1,1000,8000,'randseasonq',5,[],0.001,0.01,'amp_limited','gauss',40,0,1,2e-5,0.1,1,50],
           [1,1000,8000,'randseasonq',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,50],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,5e-5,0.1,1,50],
           [1,2000,16000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-4,0.2,1,50],
           [1,1000,8000,'randsine',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,50],
           [1,500,8000,'randsine',10,[],0.001,0.01,'amp_limited','gauss',20,0,1,5e-5,0.1,1,50],
           [1,1000,8000,'randsine',3,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-5,0.1,1,50],
           [1,1000,16000,'randsine',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,50],
           [1,1000,8000,'randsquare',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,50],
           [1,500,8000,'randsquare',10,[],0.001,0.01,'amp_limited','gauss',20,0,1,5e-5,0.1,1,50],
           [1,1000,8000,'randsquare',3,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-5,0.1,1,50],
           [1,1000,16000,'randsquare',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,2e-5,0.1,1,50],
           [1,2000,16000,'randseason',5,[],0.001,0.05,'amp_limited','gauss',20,0.2,1,2e-5,0.2,1,50],
           [1,300,8000,'randseason',5,[],0.001,0.03,'amp_limited','gauss',20,0.05,1,2e-5,0.2,1,50],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-3,0.01,1,50],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','uniform',20,0,1,2e-5,0.05,1,100],
           [1,800,8000,'randseason',5,[],0.001,0.01,'amp_limited','uniform',20,0,1,1e-5,0.02,1,100],
           [1,600,8000,'randseason',5,[],0.001,0.01,'amp_limited','uniform',20,0,1,1e-5,1,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','uniform',20,0,1,1e-5,1,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-5,0,1,100],
           [1,800,8000,'randseasonq',5,[],0.001,0.01,'amp_limited','gauss',20,0,1,1e-5,1,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0,1,1e-5,1,1,100],
           [1,800,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0,1,1e-5,0.5,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0,1,1e-5,0.5,1,100],
           [1,1000,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0,1,1e-5,0.1,1,100],
           [1,1000,8000,'randseasonq',5,[],0.001,0.1,'amp_limited','uniform',40,0,1,1e-5,0.1,1,100],
           [1,500,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0,1,2e-5,0.2,1,100],
           [1,500,8000,'randseason',5,[],0.001,0.1,'amp_limited','uniform',20,0.03,1,2e-5,0.2,1,100],
           [1,800,8000,'randseasonq',5,[],0.001,0.1,'amp_limited','uniform',60,0.03,1,5e-5,0.2,1,100]
           ]

a_config = [(0.1,('point',1),5),(0.5,('point',1),10),(1,('point',1),20),(0.5,('point',10),100),(0.1,('point',10),150),(0.1,('point',10),200),(0.3,('point',20),250),(2,('point',20),300),
            (0.1,('interval',10),320),(0.1,('interval',100),350),(0.3,('interval',100),450),(0.5,('interval',100),550),(0.2,('interval',1000),600),(0.5,('interval',500),650),
            (0.1,('contextual',10),660),(0.1,('contextual',50),680),(0.1,('contextual',200),700),(0.3,('contextual',50),750),(0.5,('contextual',10),770),(0.5,('contextual',100),800),(0.3,('contextual',400),850),(0.5,('contextual',400),900),
            (0.1,('collective',10),950),(0.1,('collective',50),1000),(0.1,('collective',100),1050),(0.3,('collective',200),1100),(0.5,('collective',300),1150),(0.5,('collective',500),1175),(0.3,('collective',1000),1200),
            (0.1,('shapelet',10),1250),(0.2,('shapelet',50),1300),(0.2,('shapelet',100),1350),(0.5,('shapelet',100),1400),(1,('shapelet',100),1450),(2,('shapelet',100),1500),(0.5,('shapelet',500),1550),
            (0.2,('noise',10),1600),(0.5,('noise',50),1700),(1,('noise',100),1800),(0.2,('noise',200),1900),(0.5,('noise',300),1950),(1,('noise',50),2000)
            ]