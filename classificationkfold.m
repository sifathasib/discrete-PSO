function accuracyy = classificationkfold(u,data)
        
    Position_vec = find(logical(u));
    
    indices = crossvalind('Kfold',data.stromaGroup,2);
    cp = classperf(data.stromaGroup); % initializes the CP object
    for i = 1:2
        test = (indices == i);train = ~test;
        
        
        Sample =data.stromaDataset(test,Position_vec) ;
        Trainer =data.stromaDataset(train,Position_vec);
        Group =data.stromaGroup(train,:) ;
        %% KNN
%         class = knnclassify(Sample,Trainer,Group);
        %% KNN Call --> 2

        class = knnclassify(Sample,Trainer,Group,2);
        %% SVM
%         SVMstruct = svmtrain(Trainer,Group,'Kernel_Function','linear');
%         class = svmclassify(SVMstruct,Sample);
        % updates the CP object with the current classification results
        classperf(cp,class,test);  
    end
    cp.CorrectRate; % queries for the correct classification rate

   accuracyy =   cp.CorrectRate *100;
end