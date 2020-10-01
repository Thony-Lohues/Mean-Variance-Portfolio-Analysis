import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize


dateparse = lambda x: pd.datetime.strptime(x,'%Y%m')
portfolios = pd.read_csv('48_Industry_Portfolios.csv', skiprows=11, nrows=1128, index_col=0, parse_dates=True, date_parser=dateparse)

'Check columns names'
portfolios.columns

'5 assets portfolio'
portfolios_5 = portfolios[['Fin  ', 'Agric', 'Aero ', 'Chems', 'Hlth ']].loc['2015-07-01':'2020-06-01']

def f_frontier(data, rf, short_selling, risk_free, n):
    
    global Z_bar_annual, Sigma_annual, Sigma_inverse_annual, A_annual, B_annual, C_annual, delta_annual, names, Sharpe_ratio, rf_annual, mu, std, w_t_, w_t
    
    names = data.columns
    Z_bar = np.mean(data)
    Z_bar_annual = ((1+Z_bar/100)**12 - 1)*100
    rf_annual = ((1+rf/100)**12-1)*100
    Sigma = np.cov(data.T)
    Sigma_annual = Sigma*12
    Sigma_inverse = np.linalg.inv(Sigma)
    Sigma_inverse_annual = np.linalg.inv(Sigma_annual)
    Ones_vec = np.ones(len(Z_bar))

    A = Ones_vec.T @ Sigma_inverse @ Ones_vec
    B = Ones_vec.T @ Sigma_inverse @ Z_bar
    C = Z_bar.T @ Sigma_inverse @ Z_bar 
    delta = A*C - B**2
    
    A_annual = Ones_vec.T @ Sigma_inverse_annual @ Ones_vec
    B_annual = Ones_vec.T @ Sigma_inverse_annual @ Z_bar_annual
    C_annual = Z_bar_annual.T @ Sigma_inverse_annual @ Z_bar_annual 
    delta_annual = A_annual*C_annual - B_annual**2
    
    if short_selling == 'oui' :
        
        if risk_free == 'non' :
            
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []
            w = np.zeros([len(mu_ss),5])

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))

            lambda_ = (C-mu_ss[np.where(std_ss == np.min(std_ss))]*B)/delta
            gamma = (mu_ss[np.where(std_ss == np.min(std_ss))]*A-B)/delta
            w_g = lambda_*Sigma_inverse@Ones_vec + gamma*Sigma_inverse@Z_bar
            z_g = Z_bar.T @ w_g
            z_g_annual = Z_bar_annual.T @ w_g
            var_g = w_g.T @ Sigma @ w_g
            var_g_annual = w_g.T @ Sigma_annual @ w_g
            print('Le poid du portefeuille a variance minimum est :', np.round(w_g,2))
            print('Le rendement mensuel du portefeuille a variance minimum est :', np.round(z_g,4))
            print('Le rendement annuel du portefeuille a variance minimum est :', np.round(z_g_annual,4))
            print('La volatilite mensuelle du portefeuille a variance minimum est :', np.round(np.sqrt(var_g),4))
            print('La volatilite annuelle du portefeuille a variance minimum est :', np.round(np.sqrt(var_g_annual),4))

    
            # Graphique 

            plt.plot(std_ss*np.sqrt(12), ((1+mu_ss/100)**12 -1)*100)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)                                          #couleur
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i]))
            plt.axis([9, 30, -10, 20])
            plt.title('Mean variance locus - short-selling/no rf')
            plt.ylabel('Return')
            plt.xlabel('σ')
            plt.plot(np.min(std_ss*np.sqrt(12)), ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100, 'x')
            plt.annotate('Min variance portfolio', (np.min(std_ss*np.sqrt(12))-7.5, ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100))
            
        
        if risk_free == 'oui' :
            
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))
                
            

            mu_ss_rf = np.linspace(-3, 3, n)
            std_ss_rf = []

            for i in range (0, len(mu_ss_rf)) :
                std_ss_rf = np.append(std_ss_rf, np.sqrt((mu_ss_rf[i]-rf)**2/(C - 2*rf*B + rf**2*A)))


            ## Graphique ##
            
            
            plt.plot(std_ss*np.sqrt(12), ((1+mu_ss/100)**12 -1)*100)
            plt.plot(std_ss_rf*np.sqrt(12), ((1+mu_ss_rf/100)**12 -1)*100)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)                                          #couleur
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i]))
            plt.axis([0, 40, -10, 20])
            plt.title('Mean variance locus - short-selling/risk-free asset')
            plt.ylabel('Return')
            plt.xlabel('σ')
            plt.plot(np.min(std_ss*np.sqrt(12)), ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100, 'x')
            plt.annotate('Min variance portfolio', (np.min(std_ss*np.sqrt(12))-13.5, ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100))
            
        
        
        # The tangency portfolio :


            w_t = (Sigma_inverse @ (Z_bar - rf * Ones_vec)) / (B - A*rf)
            Portfolio_mean = Z_bar.T @ w_t
            Portfolio_var = w_t.T@Sigma@w_t
            Portfolio_std = np.sqrt(Portfolio_var)
            Sharpe_ratio = np.round((Portfolio_mean - rf)/Portfolio_std,4)
            
            print('Les poids du portefeuilles tangent sont :', np.round(w_t,2))
            print('Le rendement espere est :', np.round(Portfolio_mean,2))
            print('La volatilite est :', np.round(Portfolio_std,2))
            print('Le ratio de Sharpe :', Sharpe_ratio)
            
            Portfolio_mean_annual = Z_bar_annual.T @ w_t
            Portfolio_var_annual = w_t.T@Sigma_annual@w_t
            Portfolio_std_annual = np.sqrt(Portfolio_var_annual)
            Sharpe_ratio_annual = np.round((Portfolio_mean_annual - rf_annual)/Portfolio_std_annual,4)
            
            print('Le rendement espere annuel est :', np.round(Portfolio_mean_annual,2))
            print('La volatilite annuelle est :', np.round(Portfolio_std_annual,2))
            print('Le ratio de Sharpe annuel :', Sharpe_ratio_annual)
    
    
    
    
    if short_selling == 'non' :
        
        if risk_free == 'non' :
            
            mu = np.linspace(np.min(Z_bar_annual),np.max(Z_bar_annual), n)
            std = []
            w_g = []
            s_r = []
            w_t_ = []

            w0 = np.ones(len(Sigma)) * 1/len(Sigma)

            bounds = [(0,1) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma_annual@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar_annual.T@w) - mu[i]}, {'type':'eq','fun': lambda w: np.sum(w)-1}), bounds = bounds).x
                std = np.append(std, np.sqrt(results_w.T@Sigma_annual@results_w))
                if i >=2 :
                    if std[i] < std[i-1] :
                        w_g = np.round(results_w,4)
                s_r = np.append(s_r,(mu[i]-rf_annual)/std[i])
                if i >= 2 :
                    if s_r[i] > s_r[i-1] :
                        w_t_ = results_w
                        
                        
            z_g_annual = Z_bar_annual.T @ w_g
            var_g_annual = w_g.T @ Sigma_annual @ w_g
            print('Le poid du portefeuille a variance minimum est :', np.round(w_g,2))
            print('Le rendement annuel du portefeuille a variance minimum est :', np.round(z_g_annual,4))
            print('La volatilite annuelle du portefeuille a variance minimum est :', np.round(np.sqrt(var_g_annual),4))

    
            ## Graphique ##
        
            mu_ss = np.linspace(-3, 3, n)
            std_ss = []
            w = np.zeros([len(mu_ss),5])

            for i in range (0, len(mu_ss)) :
                std_ss = np.append(std_ss, np.sqrt((A*mu_ss[i]**2 - 2*B*mu_ss[i] + C)/delta))
                
            plt.plot(std_ss*np.sqrt(12), ((1+mu_ss/100)**12 -1)*100)
            plt.plot(std, mu)
            cmap = plt.cm.get_cmap("hsv", len(Sigma)+1)                                          #couleur
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i], 'x', color=cmap(i))
                plt.annotate(item, (np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i]))
            plt.axis([9, 30, -10, 20])
            plt.title('Mean variance locus - short-selling/no rf')
            plt.ylabel('Return')
            plt.xlabel('σ')
            plt.plot(np.min(std_ss*np.sqrt(12)), ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100, 'x')
            plt.annotate('Min variance portfolio', (np.min(std_ss*np.sqrt(12))-7.5, ((1+mu_ss[np.argmin(std_ss*np.sqrt(12))]/100)**12 - 1)*100))
            
            plt.title('Mean variance locus - short sell/no risk free asset')
            plt.ylabel('Return')
            plt.xlabel('σ')
             
        if risk_free == 'oui' :
            
            mu_rf = np.linspace(np.min(Z_bar_annual),np.max(Z_bar_annual)*1.75, n)
            std_rf = []
            Sharpe_ratio = []
            w_t = []

            w0 = np.ones(len(Sigma)) * 1/len(Sigma)

            bounds = [(0,1000) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma_annual@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar_annual.T@w) + float((1-w.T@Ones_vec))*rf_annual - mu_rf[i]}), bounds = bounds).x
                std_rf = np.append(std_rf, np.sqrt(results_w.T@Sigma_annual@results_w))

            
            ## Graphique ##
            
            mu = np.linspace(np.min(Z_bar_annual),np.max(Z_bar_annual), n)
            std = []
            w_list = []

            w0 = np.ones(len(Sigma_annual)) * 1/len(Sigma_annual)

            bounds = [(0,1) for i in range(len(w0))]
          
            def f_variance(w):
                return 0.5*float(w.T@Sigma_annual@w)
          
            for i in range(0, n):
                results_w = minimize(f_variance, w0, method = 'SLSQP', constraints = ({'type':'eq','fun': lambda w: float(Z_bar_annual.T@w) - mu[i]}, {'type':'eq','fun': lambda w: np.sum(w)-1}), bounds = bounds).x
                std = np.append(std, np.sqrt(results_w.T@Sigma_annual@results_w))
                
            plt.plot(std, mu)
            
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i], 'x')
                plt.annotate(item, (np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i] ))  
              
            plt.plot(std_rf, mu_rf)
             
            for (i, item) in enumerate(names, start=0):
                plt.plot(np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i], 'x')
                plt.annotate(item, (np.sqrt(Sigma_annual[i,i]), Z_bar_annual[i])) 
            

            plt.title('Mean variance locus, Rf, no short sell/risk free asset')
            plt.ylabel('Return')
            plt.xlabel('σ')
          
 