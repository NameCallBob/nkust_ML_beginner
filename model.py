from sklearn.tree import DecisionTreeRegressor



class SalaryModel():
    """依照資料集，用於二元預測的薪資模型"""

    def __print_ModelScore(self,y_test,y_pred) -> list:
        """輸出模型績效指數"""
        from sklearn.metrics import confusion_matrix,precision_score,accuracy_score,recall_score,f1_score,roc_auc_score
        # 計算
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        return [cm,precision,accuracy,recall,f1,auc]
    
    def output_modelScore(self,model:list,y_test,y_pred):
        """輸出所有模型訓練之成效"""
        res = []
        for i in model:
            self.__print_ModelScore(y_test,y_pred)

    def training(self):
        """訓練"""
        pass
    
    def saveModel(self):
        """儲存已訓練之模型"""
        pass