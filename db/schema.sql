DROP TABLE IF EXISTS ChatbotTextData;
DROP TABLE IF EXISTS ChatbotUpdateHistory;

CREATE TABLE ChatbotTextData (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)) ||
                                  hex(randomblob(2)) ||
                                  '4' || substr(hex(randomblob(2)), 2) ||
                                  substr('89ab', abs(random() % 4) + 1, 1) || 
                                  substr(hex(randomblob(2)), 2) ||
                                  hex(randomblob(6)))),
    text TEXT,
    created_by TEXT,
    instance_name TEXT,
    UNIQUE(text, created_by, instance_name)
);

CREATE TABLE ChatbotUpdateHistory (
    id TEXT,
    domain_knowledge_name TEXT,
    description TEXT,
    FOREIGN KEY (id) REFERENCES ChatbotTextData(id) ON DELETE CASCADE,
    UNIQUE(id, domain_knowledge_name)
);

CREATE TRIGGER DeleteIDFromChatbotTextData
AFTER DELETE ON ChatbotUpdateHistory
BEGIN
    DELETE FROM ChatbotTextData
    WHERE id NOT IN (SELECT DISTINCT id FROM ChatbotUpdateHistory);
END;
