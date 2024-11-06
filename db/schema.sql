DROP TABLE IF EXISTS ChatbotTextData;
CREATE TABLE ChatbotTextData (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4)) ||
                                  hex(randomblob(2)) ||
                                  '4' || substr(hex(randomblob(2)), 2) ||
                                  substr('89ab', abs(random() % 4) + 1, 1) || 
                                  substr(hex(randomblob(2)), 2) ||
                                  hex(randomblob(6)))),
    text TEXT,
    created_by TEXT,
    instance_name TEXT
);
