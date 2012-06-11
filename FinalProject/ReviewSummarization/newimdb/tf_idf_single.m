%load test datartest_data(:,2)
test_data = load('imdb_test.data');
test_labels = load('imdb_test.labels');
test_docs_length = length(test_labels);
%test_docs_length = 5;

% Loading vocabulary
vocab = importdata('imdb_train_vocabulary.txt');

% for storing the top 5 important words for each document
top_imp_words = cell(1,10);

i = 10019;

    % getting data of each specific document
    doc_indices = find(test_data(:,1) == i);
    document = test_data(doc_indices,:);
    
    %Calculating tf's for each term in the document
    no_of_terms = size(document,1);
    tf_doc = document(:,3)/no_of_terms;
    
    idf_doc = zeros(no_of_terms,1);
    for j = 1:no_of_terms
        indices = find(test_data(:,2) == document(j,2));
        doc_ids = test_data(indices,1);
        doc_freq = length(doc_ids);
        idf_doc(j) = log(length(test_labels)/doc_freq);
    end
    
    tf_idf_weights = tf_doc .* idf_doc;
    
    [sortedValues,sortIndex] = sort(tf_idf_weights(:),'descend');
    if 5 < length(sortIndex)
        maxIndices = sortIndex(1:10);
    end
    
    
    top_imp_words(1,:) = vocab(document(maxIndices,2));
    
