// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Parser
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ir.hpp"
#include <istream>
#include <string>
#include <unordered_map>

namespace pto::wsp::ir {

// Parse errors
class ParseError : public std::runtime_error {
public:
    int line;
    int column;
    std::string message;

    ParseError(int line_, int column_, std::string msg)
        : std::runtime_error(formatError(line_, column_, msg)),
          line(line_), column(column_), message(std::move(msg)) {}

private:
    static std::string formatError(int line, int col, const std::string& msg) {
        return "Parse error at line " + std::to_string(line) +
               ", column " + std::to_string(col) + ": " + msg;
    }
};

// Token types
enum class TokenType {
    Eof,
    Newline,
    Ident,
    String,
    Number,
    At,        // @
    Colon,     // :
    Comma,     // ,
    LParen,    // (
    RParen,    // )
    LBrace,    // {
    RBrace,    // }
    LBracket,  // [
    RBracket,  // ]
    Equals,    // =
    Pipe,      // |
    Percent,   // %
    Exclaim,   // !

    // Keywords
    KwModule, KwVersion, KwTarget,
    KwWorkload, KwSchedule, KwPipeline,
    KwParallelFor, KwForEach, KwSelect, KwCond, KwTask,
    KwCombine, KwSequential, KwCall,
    KwChannel, KwProcess, KwSend, KwConsume,
    KwDispatch, KwStreams, KwStreamBy, KwTiming,
    KwSpatialMap, KwLayout,
    KwDense, KwDenseDyn, KwRagged, KwSparse,
    KwIn, KwFor, KwAs, KwElse, KwYield, KwResources,
    KwConsumes, KwProduces,

    // Schedule policy keywords
    KwRoundRobin, KwAffinity, KwHash, KwWorkSteal,
    KwImmediate, KwBatched, KwInterleaved, KwRateLimit,
    KwPipelineDepth, KwTaskWindow, KwBatchDeps,
};

// Token
struct Token {
    TokenType type;
    std::string value;
    int line;
    int column;
};

// Lexer - tokenizes input stream
class Lexer {
public:
    explicit Lexer(std::istream& is);
    Token next();

private:
    std::istream& is_;
    int line_ = 1;
    int column_ = 1;
    int current_ = -1;

    int advance();
    int peek();
    void skipWhitespace();
    void skipLineComment();
    void consumeChar();

    static const std::unordered_map<std::string, TokenType>& keywords();
};

// Parser - parses token stream into IR
class Parser {
public:
    explicit Parser(std::istream& is);

    // Parse a complete module
    Module parse();

    // Convenience methods
    static Module parseString(const std::string& source);
    static Module parseFile(const std::string& path);

private:
    Lexer lexer_;
    Token current_;
    IRFactory factory_;

    Token advance();
    [[nodiscard]] bool check(TokenType type) const;
    Token expect(TokenType type, const std::string& msg);
    bool match(TokenType type);

    // Parsing methods
    IRPtr<AxisNode> parseAxis();
    IRPtr<WorkloadNode> parseWorkload();
    IRPtr<WorkloadNode> parseWorkloadBody();
    IRPtr<WorkloadNode> parseStatement();
    WorkloadDef parseWorkloadDef();
    ScheduleDef parseScheduleDef();
    IRPtr<ScheduleNode> parseScheduleDirective();
    void parseHeader(Module& m);
};

}  // namespace pto::wsp::ir
